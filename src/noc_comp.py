'''
==================================================
File: noc_comp.py
Project: ReNOS
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati, Jakub Jastrzebski (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
NoC comparison script: compares fast analytical model with Booksim2 simulator
across different partitioning configurations.
"""

from tabnanny import verbose
from graph import model_to_graph
from utils.partitioner_utils import search_space_split_factors
from graph import TaskGraph
from models import *
import graph as dg
from graph import model_to_graph
import domain as dm
import mapper as mp
from utils.partitioner_utils import *
from utils.plotting_utils import *
import mapper as ma
import simulator_stub as ss
import fast_analytical_simulator_stub as ssfam
from visualizer import plot_timeline
from utils.ani_utils import visualize_simulation
from utils.model_fusion import fuse_conv_bn
from latency_energy_analysis import export_simulation_results_to_csv
import time
import csv
import os
import json
import pandas as pd
import argparse


def count_operational_layers(model):
    """Count layers that need partitioning tuples"""
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ not in ['InputLayer']:
            count += 1
    return count


def build_partitioner_tuples_for_model(model, layer_configs_dict, verbose=False):
    """
    Build complete partitioner tuples list for model, automatically handling
    InputLayer and Add layers with (0,1,1).

    Args:
        model: Keras model
        layer_configs_dict: Dict mapping layer_id -> [spatial, output, input_split]
                          Layer IDs correspond to operational layers (excluding InputLayer)

    Returns:
        List of tuples for each layer in the model
    """
    partitioner_tuples = []
    operational_layer_idx = 0  # Counter for operational layers (for JSON mapping)

    for layer_idx, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__

        if layer_type == 'InputLayer':
            # Always use (0,1,1) for InputLayer
            partitioner_tuples.append((0, 1, 1))
            if verbose:
                print(f"  Layer {layer_idx} ({layer.name}): InputLayer -> (0, 1, 1)")

        elif layer_type == 'Add':
            # Automatically use (0,1,1) for Add layers
            partitioner_tuples.append((0, 1, 1))
            if verbose:
                print(f"  Layer {layer_idx} ({layer.name}): Add -> (0, 1, 1) [auto]")
            operational_layer_idx += 1

        else:
            # Regular operational layer - get from config
            operational_layer_idx += 1
            layer_key = str(operational_layer_idx)

            if layer_key in layer_configs_dict:
                config = layer_configs_dict[layer_key]
                if len(config) == 3:
                    partitioner_tuples.append(tuple(config))
                    if verbose:
                        print(f"  Layer {layer_idx} ({layer.name}): {layer_type} -> {tuple(config)}")
                else:
                    raise ValueError(f"Layer {operational_layer_idx} config must have 3 values, got {config}")
            else:
                raise ValueError(f"Missing config for operational layer {operational_layer_idx} ({layer.name})")

    return partitioner_tuples


def load_partitioner_tuples_from_json(json_path, model):
    """
    Load per-layer partitioning configuration from JSON file.
    Automatically handles InputLayer and Add layers with (0,1,1).

    Args:
        json_path: Path to JSON file with format: {"1": [spatial, output, input], "2": [...], ...}
                    Layer IDs correspond to operational layers (excluding InputLayer)
        model: Keras model to build tuples for

    Returns:
        List of tuples for each layer in the model
    """
    with open(json_path, 'r') as f:
        layer_configs = json.load(f)

    print(f"  Found layer IDs in config: {sorted([int(k) for k in layer_configs.keys()])}")
    print(f"  Building partitioner tuples (Add layers auto-set to (0,1,1)):")

    return build_partitioner_tuples_for_model(model, layer_configs)


def load_sweep_configurations(sweep_json_path, model):
    """
    Load multiple configurations from a sweep JSON file.

    Args:
        sweep_json_path: Path to sweep JSON file with format:
                        {
                          "sweep_metadata": {...},
                          "configurations": [
                            {
                              "metric": "flops",
                              "target_value": 100000,
                              "avg_flops": ...,
                              "configuration": {"1": [s, o, i], ...}
                            },
                            ...
                          ]
                        }
        model: Keras model to build tuples for

    Returns:
        List of tuples: [(config_name, partitioner_tuples, metadata), ...]
    """
    with open(sweep_json_path, 'r') as f:
        sweep_data = json.load(f)

    if 'configurations' not in sweep_data:
        raise ValueError(f"Invalid sweep file format. Expected 'configurations' key in {sweep_json_path}")

    configurations = sweep_data['configurations']
    sweep_metadata = sweep_data.get('sweep_metadata', {})

    print(f"Loading sweep file: {sweep_json_path}")
    print(f"  Metric: {sweep_metadata.get('metric', 'unknown')}")
    print(f"  Total configurations in file: {len(configurations)}")

    result = []
    for idx, config in enumerate(configurations):
        layer_configs = config['configuration']
        partitioner_tuples = build_partitioner_tuples_for_model(model, layer_configs, verbose=False)

        # Create descriptive name with metadata
        metric = config.get('metric', 'unknown')
        target = config.get('target_value', 0)
        avg_flops = config.get('avg_flops', 0)
        avg_size = config.get('avg_size', 0)

        config_name = f"sweep_{metric}_{int(target)}"
        metadata = {
            'sweep_file': sweep_json_path,
            'config_index': idx,
            'metric': metric,
            'target_value': target,
            'avg_flops': avg_flops,
            'avg_size': avg_size,
            'avg_flops_no_relu': config.get('avg_flops_no_relu', 0),
            'avg_size_no_relu': config.get('avg_size_no_relu', 0)
        }

        result.append((config_name, partitioner_tuples, metadata))

    print(f"  Loaded {len(result)} configurations\n")
    return result


if __name__ == '__main__':
    # Start timing
    script_start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='NoC comparison with per-layer partitioning configurations')
    parser.add_argument('--config-files', nargs='+', default=None,
                        help='One or more JSON files with per-layer partitioning configurations')
    parser.add_argument('--max-num-partitions-for-fixed-per-layer', type=int, default=None,
                        help='Maximum number of partitions per layer for generating fixed configurations (integer).')
    parser.add_argument('--combinations-to-test', type=int, default=None,
                        help='Number of combinations to test for fixed per-layer configurations (integer).')
    parser.add_argument('--batch', type=str, default=None,
                        help='Run in batch mode. Format: "X/N" where X is batch number (1-indexed) and N is total batches. Example: --batch 2/4')
    args = parser.parse_args()

    # Parse batch argument
    batch_num = None
    total_batches = None
    if args.batch:
        try:
            batch_parts = args.batch.split('/')
            batch_num = int(batch_parts[0])
            total_batches = int(batch_parts[1])
            if batch_num < 1 or batch_num > total_batches:
                raise ValueError(f"Batch number must be between 1 and {total_batches}")
            print(f"Running in batch mode: batch {batch_num} of {total_batches}\n")
        except (ValueError, IndexError) as e:
            print(f"Error: Invalid --batch format. Use 'X/N' (e.g., --batch 2/4)")
            print(f"Details: {e}")
            exit(1)

    # Model setup
    model = ResNet_block_smaller((52, 52, 32), verbose=True)
    model = fuse_conv_bn(model, verbose=True)

    model_name = "ResNet_block_smaller"
    result_file = "./data/noc_comp_flops_sizes_29Oct"

    # Determine suffix based on mode
    if args.config_files:
        # Detect suffix from config filename pattern
        first_config = args.config_files[0]
        if 'config_flops' in first_config or 'config_flop' in first_config:
            suffix = "flops"
        elif 'config_size' in first_config:  # Matches both 'config_size' and 'config_sizes'
            suffix = "sizes"
        else:
            suffix = "custom"
    elif args.max_num_partitions_for_fixed_per_layer:
        suffix = "fixed_partitions"
    else:
        suffix = "default"

    print(f"Running in mode: {suffix}\n")

    # Prepare directories
    os.makedirs(result_file, exist_ok=True)
    jsons_dir = os.path.join(result_file, "jsons")
    os.makedirs(jsons_dir, exist_ok=True)
    layer_traffic_dir = os.path.join(result_file, "layer_traffic")
    os.makedirs(layer_traffic_dir, exist_ok=True)

    # Prepare CSV files with suffix and batch prefix
    if args.batch:
        batch_prefix = f"batch{batch_num}_"
    else:
        batch_prefix = ""

    csv_filename = f"{result_file}/{batch_prefix}{suffix}_noc_comparison_{model_name}.csv"
    energy_csv_filename = f"{result_file}/{batch_prefix}{suffix}_latency_energy_{model_name}.csv"

    # Grid setup
    x_of_grid = 12
    source = 0
    drain = 143
    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source=source, drain=drain)

    # Get number of layers that need tuples
    num_layers = count_operational_layers(model)
    print(f"\nModel has {num_layers} operational layers requiring tuples\n")

    # CSV header
    fieldnames = [
        'num_partitions',
        'parts_per_layer',
        'result_analytical',
        'result_booksim',
        'percentage_diff',
        'analytical_time',
        'booksim_time',
        'time_gain',
        'partitioner_config',
        'config_file',
        'metric',
        'target_value',
        'avg_flops',
        'avg_size',
        'avg_flops_no_relu',
        'avg_size_no_relu'
    ]
    
    # Write header first
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Load configurations from JSON files or use default
    configurations_list = []
    if args.config_files:
        print(f"Loading configurations from {len(args.config_files)} JSON file(s)...\n")
        for config_file in args.config_files:
            try:
                # Try to detect if this is a sweep file or single config file
                with open(config_file, 'r') as f:
                    data = json.load(f)

                # Check if it's a sweep file (has 'configurations' key)
                if 'configurations' in data:
                    print(f"Detected sweep file: {config_file}")
                    sweep_configs = load_sweep_configurations(config_file, model)

                    # Convert to format compatible with existing code
                    for config_name, partitioner_tuples, metadata in sweep_configs:
                        # Store metadata in a tuple with the config
                        configurations_list.append((config_name, partitioner_tuples, metadata))

                    print(f"✓ Loaded {len(sweep_configs)} configurations from sweep file\n")
                else:
                    # Single configuration file
                    print(f"Detected single config file: {config_file}")
                    partitioner_tuples = load_partitioner_tuples_from_json(config_file, model)
                    # Add empty metadata dict for compatibility
                    configurations_list.append((config_file, partitioner_tuples, {}))
                    print(f"✓ Loaded {config_file}\n")

            except Exception as e:
                print(f"✗ Error loading {config_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Apply batch filtering if in batch mode
        if args.batch:
            total_configs = len(configurations_list)
            configs_per_batch = total_configs // total_batches
            remainder = total_configs % total_batches

            # Calculate start and end indices for this batch
            # Distribute remainder across first batches
            if batch_num <= remainder:
                start_idx = (batch_num - 1) * (configs_per_batch + 1)
                end_idx = start_idx + configs_per_batch + 1
            else:
                start_idx = remainder * (configs_per_batch + 1) + (batch_num - remainder - 1) * configs_per_batch
                end_idx = start_idx + configs_per_batch

            configurations_list = configurations_list[start_idx:end_idx]
            print(f"Batch {batch_num}/{total_batches}: Running configs {start_idx+1} to {end_idx} (out of {total_configs} total)\n")
            
    elif args.max_num_partitions_for_fixed_per_layer:

        combination_int = args.combinations_to_test if args.combinations_to_test else 3
        max_number_of_parts = args.max_num_partitions_for_fixed_per_layer
        print(f"Generating fixed per-layer configurations with max {max_number_of_parts} partitions...\n")

        # Generate configurations with fixed per-layer partitions
        fixed_configs = [(i, j, k) for i in range(0, combination_int)
                        for j in range(1, combination_int)
                        for k in range(1, combination_int)]
        num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in fixed_configs]
        
        print(f"Generated {len(fixed_configs)} configurations\n")

        # Filter combination to not exceed max partitions
        fixed_configs = [cfg for cfg, num_parts in zip(fixed_configs, num_partitions_per_layer)
                        if num_parts <= max_number_of_parts]

        print(f"After filtering: {len(fixed_configs)} configurations\n")

        # Apply batch filtering if in batch mode
        batch_start_idx = 0  # Track the global starting index for this batch
        if args.batch:
            total_configs = len(fixed_configs)
            configs_per_batch = total_configs // total_batches
            remainder = total_configs % total_batches

            # Calculate start and end indices for this batch
            if batch_num <= remainder:
                start_idx = (batch_num - 1) * (configs_per_batch + 1)
                end_idx = start_idx + configs_per_batch + 1
            else:
                start_idx = remainder * (configs_per_batch + 1) + (batch_num - remainder - 1) * configs_per_batch
                end_idx = start_idx + configs_per_batch

            batch_start_idx = start_idx  # Remember where this batch starts globally
            fixed_configs = fixed_configs[start_idx:end_idx]
            print(f"Batch {batch_num}/{total_batches}: Running configs {start_idx+1} to {end_idx} (out of {total_configs} total)\n")

        print(" These are the configurations to be tested:")
        print( '--------------------------------------------')
        print(fixed_configs)
        print( '--------------------------------------------')

        # Build partitioner tuples for each configuration
        for idx, config in enumerate(fixed_configs):
            # Create config dict where all layers use the same tuple
            config_dict = {str(i): list(config) for i in range(1, num_layers + 1)}

            # Build tuples (handles Add layers automatically)
            partitioner_tuples = build_partitioner_tuples_for_model(model, config_dict)

            # Use global index for config naming (batch_start_idx + idx + 1)
            global_config_num = batch_start_idx + idx + 1

            verbose = False
            if verbose:
                print(f"Configuration {global_config_num}: {config} ({2**config[0] * config[1] * config[2]} parts/layer)")

            configurations_list.append((f"fixed_config_{global_config_num}", partitioner_tuples, {}))
            
    else:
        # Default configuration - build with (2,2,2) for regular layers, (0,1,1) for Add
        print("No config files provided. Using default configuration: (2,2,2) for operational layers\n")

        # Create default config dict with (2,2,2) for all operational layers
        default_config = {}
        for i in range(1, num_layers + 1):
            default_config[str(i)] = [2, 2, 2]

        partitioner_tuples = build_partitioner_tuples_for_model(model, default_config)
        configurations_list.append(("default", partitioner_tuples, {}))

    ##############################################################################
    # Run simulations for each configuration
    ##############################################################################
    
    print(f"\nRunning {len(configurations_list)} configuration(s)\n")
            
    # Iterate over configurations
    iteration = 0
    for config_data in configurations_list:
        # Unpack configuration data (handle both 2-tuple and 3-tuple formats)
        if len(config_data) == 3:
            config_name, partitioner_tuples, metadata = config_data
        else:
            config_name, partitioner_tuples = config_data
            metadata = {}

        iteration += 1
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{len(configurations_list)}: {config_name}")
        if metadata:
            print(f"Metadata: metric={metadata.get('metric')}, target={metadata.get('target_value'):.2e}, "
                  f"avg_flops={metadata.get('avg_flops', 0):.2e}, avg_size={metadata.get('avg_size', 0):,.0f}")
        print(f"Partitioner tuples: {partitioner_tuples}")
        print(f"{'='*80}\n")

        try:
            # Calculate number of partitions
            num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in partitioner_tuples]
            total_partitions = sum(num_partitions_per_layer)
            print(f"Used partitioner tuples: {partitioner_tuples}")
            print(f"Total partitions: {total_partitions}")

            # Build task graph
            dep_graph = TaskGraph(source=grid.source, drain=grid.drain)
            parts, deps = build_partitions_splitting_input_for_many_tuples(
                model, grid, partitioning_tuple=partitioner_tuples,
                grouping=False, verbose=False
            )

            # Compute layer connection metrics from actual dependencies
            connection_metrics = compute_layer_connection_metrics(parts, deps, verbose=False)
            print("\nConnection Metrics per Layer:")
            for layer_name, metrics in connection_metrics.items():
                print(f"  {layer_name}: "
                    f"in={metrics['input_connections']}, "
                    f"out={metrics['output_connections']}, "
                    f"partitions={metrics['num_partitions']}")

            # Compute per-layer traffic density metrics
            # Build partitioning config dict for better reporting
            partitioning_configs = {}
            for layer_idx, layer in enumerate(model.layers):
                if layer_idx < len(partitioner_tuples):
                    partitioning_configs[layer.name] = partitioner_tuples[layer_idx]

            traffic_density_metrics = compute_layer_traffic_density_metrics(
                parts, deps,
                partitioning_configs=partitioning_configs,
                verbose=True
            )

            # Export traffic density to separate CSV per configuration
            traffic_df = pd.DataFrame(traffic_density_metrics)

            # Add configuration info to each row
            traffic_df['model_config'] = str(partitioner_tuples[1]) if len(partitioner_tuples) > 1 else str(partitioner_tuples[0])
            traffic_df['total_partitions'] = total_partitions

            # Extract config basename (filename without path and .json extension)
            if config_name == "default" or config_name.startswith("fixed_config_"):
                config_basename = config_name
            else:
                config_basename = os.path.basename(config_name).replace('.json', '')

            # Save to separate file per configuration
            per_config_traffic_csv = os.path.join(layer_traffic_dir, f"{config_basename}_layer_traffic_density_{model_name}.csv")
            traffic_df.to_csv(per_config_traffic_csv, index=False)
            print(f"✓ Saved layer traffic density to: {per_config_traffic_csv}")

            # Calculate mean traffic density across all layers for energy CSV
            mean_in_traffic_density = traffic_df['mean_in_traffic_density_bytes'].mean()
            mean_out_traffic_density = traffic_df['mean_out_traffic_density_bytes'].mean()

            task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
            path = row_wise_mapping(task_graph, grid, verbose=False)

            # Construct mapping
            mapping = {task_id: int(next_node) for task_id, _, next_node in path
                        if task_id != "start" and task_id != "end"}

            # Generate unique config filename in jsons/ directory
            config_basename = os.path.basename(config_name).replace('.json', '') if config_name != "default" else "default"
            config_path_file = os.path.join(jsons_dir, f"mapping_{config_basename}_iter{iteration}.json")
            
            mapper_config = "." + config_path_file
            mapper = ma.Mapper()
            mapper.init(task_graph, grid)
            mapper.set_mapping(mapping)
            mapper.mapping_to_json(mapper_config,
                                    file_to_append="./config_files/arch.json")

            # Run Fast Analytical model
            print("Running Fast Analytical model simulation...")
            fast_sim = ssfam.FastAnalyticalSimulatorStub()
            start_time = time.time()
            result_fast_anal, logger_fast_anal = fast_sim.run_simulation(
                config_path_file, verbose=False
            )
            fast_analytical_time = time.time() - start_time
            print(f"  Result: {result_fast_anal} cycles")
            print(f"  Time: {fast_analytical_time:.4f} seconds")
            
            # Read config and enable logger/sim_power
            with open(config_path_file, 'r') as f:
                config = json.load(f)

            # Ensure logger and sim_power are enabled
            if 'arch' in config:
                config['arch']['logger'] = 1
                config['arch']['sim_power'] = 1

                # Write back the modified config
                with open(config_path_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print("  ✓ Enabled logger and sim_power in config")
            else:
                print("  ✗ Config file missing 'arch' section. Skipping logger setup.")

            # Run Booksim2 simulation
            print("\nRunning Booksim2 simulation...")
            stub = ss.SimulatorStub()
            start_time = time.time()
            result_booksim, logger = stub.run_simulation(config_path_file, dwrap=False)
            booksim_time = time.time() - start_time
            print(f"  Result: {result_booksim} cycles")
            print(f"  Time: {booksim_time:.4f} seconds")

            # Export latency and energy analysis to CSV
            print("\nExporting latency/energy analysis...")
            try:
                # In batch mode, always append within this batch's CSV
                # Non-batch mode: append after first iteration
                should_append = (iteration > 1)

                csv_df = export_simulation_results_to_csv(
                    logger=logger,
                    config_path=config_path_file,
                    output_path=energy_csv_filename,
                    append=should_append,
                    num_partitions=total_partitions,
                    parts_per_layer=num_partitions_per_layer[1],
                    partitioner_config=str(partitioner_tuples[1]),
                    connection_metrics=connection_metrics,
                    mean_in_traffic_density=mean_in_traffic_density,
                    mean_out_traffic_density=mean_out_traffic_density
                )
                print(f"  ✓ Energy analysis saved to: {energy_csv_filename}")

                # Print energy summary
                print(f"\n  Energy Summary:")
                print(f"    PE Energy: {csv_df['energy_PEs_uJ'].values[0]:.3f} µJ")
                print(f"    Data Flow Energy: {csv_df['energy_data_flow_uJ'].values[0]:.3f} µJ")
                print(f"    Total Energy: {csv_df['total_energy_uJ'].values[0]:.3f} µJ")
            except Exception as e:
                print(f"  ✗ Failed to export energy analysis: {e}")

            # Calculate metrics
            percentage_diff = abs(result_fast_anal - result_booksim) / result_booksim * 100
            time_gain = booksim_time / fast_analytical_time

            print(f"\nComparison:")
            print(f"  Difference: {abs(result_fast_anal - result_booksim)} cycles ({percentage_diff:.2f}%)")
            print(f"  Time gain: {time_gain:.4f}x")

            # Write to CSV immediately after each iteration
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'num_partitions': total_partitions,
                    'parts_per_layer': num_partitions_per_layer[1],
                    'result_analytical': result_fast_anal,
                    'result_booksim': result_booksim,
                    'percentage_diff': f"{percentage_diff:.2f}",
                    'analytical_time': f"{fast_analytical_time:.4f}",
                    'booksim_time': f"{booksim_time:.4f}",
                    'time_gain': f"{time_gain:.4f}",
                    'partitioner_config': str(partitioner_tuples[1:]),
                    'config_file': config_basename,
                    'metric': metadata.get('metric', ''),
                    'target_value': metadata.get('target_value', ''),
                    'avg_flops': f"{metadata.get('avg_flops', 0):.2e}" if metadata.get('avg_flops') else '',
                    'avg_size': f"{metadata.get('avg_size', 0):.0f}" if metadata.get('avg_size') else '',
                    'avg_flops_no_relu': f"{metadata.get('avg_flops_no_relu', 0):.2e}" if metadata.get('avg_flops_no_relu') else '',
                    'avg_size_no_relu': f"{metadata.get('avg_size_no_relu', 0):.0f}" if metadata.get('avg_size_no_relu') else ''
                })

            print(f"✓ Results appended to {csv_filename}")

        except Exception as e:
            print(f"\n✗ Error with config {config_name}: {str(e)}")
            print(f"  Skipping to next iteration...\n")
            continue

    # Calculate total execution time
    script_end_time = time.time()
    total_execution_time = script_end_time - script_start_time

    # Convert to human-readable format
    hours = int(total_execution_time // 3600)
    minutes = int((total_execution_time % 3600) // 60)
    seconds = total_execution_time % 60

    print(f"\n{'='*80}")
    if args.batch:
        print(f"BATCH {batch_num}/{total_batches} COMPLETE")
    else:
        print(f"SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  - {csv_filename}")
    print(f"  - {energy_csv_filename}")
    print(f"  - {layer_traffic_dir}/ (per-config traffic density)")
    print(f"\nConfigurations tested in this {'batch' if args.batch else 'run'}: {len(configurations_list)}")
    print(f"Total execution time: {hours}h {minutes}m {seconds:.2f}s")
    if args.batch:
        print(f"\nNote: This is batch {batch_num} of {total_batches}. Run other batches separately.")
    print(f"{'='*80}\n")
        









