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


def build_partitioner_tuples_for_model(model, layer_configs_dict):
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
            print(f"  Layer {layer_idx} ({layer.name}): InputLayer -> (0, 1, 1)")

        elif layer_type == 'Add':
            # Automatically use (0,1,1) for Add layers
            partitioner_tuples.append((0, 1, 1))
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


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='NoC comparison with per-layer partitioning configurations')
    parser.add_argument('--config-files', nargs='+', default=None,
                        help='One or more JSON files with per-layer partitioning configurations')
    args = parser.parse_args()

    # Model setup
    model = ResNet_block_smaller((52, 52, 32), verbose=True)
    model = fuse_conv_bn(model, verbose=True)

    model_name = "ResNet_block_smaller"
    result_file = "./data/noc_comp_flops_sizes_29Oct"

    # Prepare CSV files
    csv_filename = f"{result_file}/noc_comparison_results_{model_name}.csv"
    energy_csv_filename = f"{result_file}/latency_energy_results_{model_name}.csv"
    traffic_density_csv_filename = f"{result_file}/layer_traffic_density_{model_name}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

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
        'config_file'
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
                partitioner_tuples = load_partitioner_tuples_from_json(config_file, model)
                configurations_list.append((config_file, partitioner_tuples))
                print(f"✓ Loaded {config_file}\n")
            except Exception as e:
                print(f"✗ Error loading {config_file}: {e}")
                continue
    else:
        # Default configuration - build with (2,2,2) for regular layers, (0,1,1) for Add
        print("No config files provided. Using default configuration: (2,2,2) for operational layers\n")

        # Create default config dict with (2,2,2) for all operational layers
        default_config = {}
        for i in range(1, num_layers + 1):
            default_config[str(i)] = [2, 2, 2]

        partitioner_tuples = build_partitioner_tuples_for_model(model, default_config)
        configurations_list.append(("default", partitioner_tuples))

    print(f"\nRunning {len(configurations_list)} configuration(s)\n")

    breakpoint()
    
    # Iterate over configurations
    iteration = 0
    for config_name, partitioner_tuples in configurations_list:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{len(configurations_list)}: {config_name}")
        print(f"Partitioner tuples: {partitioner_tuples}")
        print(f"{'='*80}\n")

        try:
            # Calculate number of partitions
            num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in partitioner_tuples]
            total_partitions = sum(num_partitions_per_layer)
            print(f"Used partitioner tuples: {partitioner_tuples[2]}")
            print(f"Total partitions: {total_partitions}")
            print(f"Partitions per layer: {num_partitions_per_layer}\n")

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

            # Export traffic density to CSV
            traffic_df = pd.DataFrame(traffic_density_metrics)

            # Add configuration info to each row
            traffic_df['model_config'] = str(partitioner_tuples[1]) if len(partitioner_tuples) > 1 else str(partitioner_tuples[0])
            traffic_df['total_partitions'] = total_partitions

            # Write or append to CSV
            if iteration == 1:
                traffic_df.to_csv(traffic_density_csv_filename, index=False)
                print(f"✓ Created traffic density CSV: {traffic_density_csv_filename}")
            else:
                traffic_df.to_csv(traffic_density_csv_filename, mode='a', header=False, index=False)
                print(f"✓ Appended to traffic density CSV")

            task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
            path = row_wise_mapping(task_graph, grid, verbose=False)

            # Construct mapping
            mapping = {task_id: int(next_node) for task_id, _, next_node in path
                        if task_id != "start" and task_id != "end"}

            # Generate unique config filename based on iteration or config name
            config_basename = os.path.basename(config_name).replace('.json', '') if config_name != "default" else "default"
            config_path_file = result_file + f"/mapping_{config_basename}_iter{iteration}.json"
            
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
                csv_df = export_simulation_results_to_csv(
                    logger=logger,
                    config_path=config_path_file,
                    output_path=energy_csv_filename,
                    append=(iteration > 1),  # Append after first iteration
                    num_partitions=total_partitions,
                    parts_per_layer=num_partitions_per_layer[1],
                    partitioner_config=str(partitioner_tuples[1]),
                    connection_metrics=connection_metrics
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
                    'result_analytical': result_fast_anal,
                    'result_booksim': result_booksim,
                    'percentage_diff': f"{percentage_diff:.2f}",
                    'analytical_time': f"{fast_analytical_time:.4f}",
                    'booksim_time': f"{booksim_time:.4f}",
                    'time_gain': f"{time_gain:.4f}",
                    'partitioner_config': str(partitioner_tuples[1:]),
                    'config_file': config_name
                })

            print(f"✓ Results appended to {csv_filename}")

        except Exception as e:
            print(f"\n✗ Error with config {config_name}: {str(e)}")
            print(f"  Skipping to next iteration...\n")
            continue

    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_filename}")
    print(f"{'='*80}\n")
        









