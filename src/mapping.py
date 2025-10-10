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
import argparse


def count_operational_layers(model):
    """Count layers that need partitioning tuples"""
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ not in ['InputLayer']:
            count += 1
    return count


def get_model(model_name, verbose=True):
    """Load model based on name"""
    models_config = {
        'AlexNet': (AlexNet, (32, 32, 3), {'num_classes': 10, 'verbose': verbose}),
        'VGG_16_early': (VGG_16_early_layers, (32, 32, 3), {'num_classes': 10, 'verbose': verbose}),
        'VGG_16_late': (VGG_16_late_layers, (4, 4, 256), {'num_classes': 10, 'verbose': verbose}),
        'MobileNetv1': (MobileNetv1, (32, 32, 3), {'num_classes': 10, 'verbose': verbose}),
        'ResNet32_early': (ResNet32_early_blocks, (32, 32, 3), {'verbose': verbose}),
        'ResNet32_mid': (ResNet32_mid_blocks, (32, 32, 16), {'num_classes': 10, 'verbose': verbose}),
        'ResNet32_late': (ResNet32_late_blocks, (16, 16, 32), {'num_classes': 10, 'verbose': verbose}),
    }

    if model_name not in models_config:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_config.keys())}")

    model_class, input_shape, kwargs = models_config[model_name]
    model = model_class(input_shape, **kwargs)
    model = fuse_conv_bn(model, verbose=verbose)
    return model


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run mapping simulations with configurable parameters')
    parser.add_argument('-model', '--model', type=str, required=True,
                        choices=['AlexNet', 'VGG_16_early', 'VGG_16_late', 'MobileNetv1',
                                'ResNet32_early', 'ResNet32_mid', 'ResNet32_late'],
                        help='Model to use for simulation')
    parser.add_argument('-config', '--config', type=int, nargs=3, required=True,
                        metavar=('SPATIAL', 'OUTPUT', 'INPUT_SPLIT'),
                        help='Configuration tuple: spatial output input_split')
    parser.add_argument('-mapping', '--mapping', type=str, required=True,
                        choices=['random', 'row_wise', 'column_wise'],
                        help='Mapping strategy to use')
    parser.add_argument('-runs', '--runs', type=int, default=10,
                        help='Number of runs (only for random mapping, default: 10)')
    parser.add_argument('-run_booksim', '--run_booksim', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Run Booksim2 simulation (True/False, default: False)')

    args = parser.parse_args()

    # Load model
    print(f"\n{'='*80}")
    print(f"Loading model: {args.model}")
    print(f"{'='*80}\n")
    model = get_model(args.model, verbose=True)
    model_name = args.model
    result_file = "./data/mapping_comparison"

    # Prepare CSV files
    csv_filename = f"{result_file}/mapping_comparison_{model_name}.csv"
    #energy_csv_filename = f"{result_file}/latency_energy_{model_name}.csv"
    os.makedirs(result_file, exist_ok=True)

    # Grid setup
    x_of_grid = 12
    source = 0
    drain = 143
    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source=source, drain=drain)

    # Get number of layers that need tuples
    num_layers = count_operational_layers(model)
    print(f"\nModel has {num_layers} operational layers requiring tuples\n")

    # Configuration from command line
    configuration = tuple(args.config)  # (spatial, output, input_split)

    # Determine number of runs (for random mapping)
    num_runs = args.runs if args.mapping == 'random' else 1

    # CSV header - always include all fields
    fieldnames = [
        'mapping_strategy',
        'num_partitions',
        'parts_per_layer',
        'total_tasks',
        'result_analytical',
        'analytical_time',
        'partitioner_config',
        'result_booksim',
        'booksim_time',
        'percentage_diff',
        'time_gain'
    ]

    # Check if CSV exists, if not write header
    csv_exists = os.path.isfile(csv_filename)
    if not csv_exists:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Created new CSV file: {csv_filename}\n")
    else:
        print(f"Appending to existing CSV file: {csv_filename}\n")

    spatial, output, input_split = configuration
    print(f"\n{'='*80}")
    print(f"Testing Configuration: (spatial={spatial}, output={output}, input={input_split})")
    print(f"{'='*80}\n")

    # Build partitioner tuples: first tuple is (0,1,1), rest are (spatial, output, input_split)
    partitioner_tuples = [(0, 1, 1)] + [(spatial, output, input_split)] * (num_layers)

    # Calculate number of partitions
    num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in partitioner_tuples]
    total_partitions = sum(num_partitions_per_layer)
    print(f"Partitioner tuples: {partitioner_tuples[1]}")
    print(f"Total partitions: {total_partitions}")
    print(f"Partitions per layer: {num_partitions_per_layer}\n")

    # Build task graph (only once, same for all mapping strategies)
    dep_graph = TaskGraph(source=grid.source, drain=grid.drain)
    parts, deps = build_partitions_splitting_input_for_many_tuples(
        model, grid, partitioning_tuple=partitioner_tuples,
        grouping=False, verbose=False
    )

    task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
    total_tasks = task_graph.n_nodes
    print(f"Total tasks in graph: {total_tasks}\n")

    # Create simulator stubs once (reuse for all mappings)
    fast_sim = ssfam.FastAnalyticalSimulatorStub()
    booksim_stub = ss.SimulatorStub()

    # Run simulations (multiple times for random mapping)
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        if args.mapping == 'random':
            print(f"[{run_idx + 1}/{num_runs}] Testing Mapping Strategy: {args.mapping.upper()} (Run {run_idx + 1})")
        else:
            print(f"Testing Mapping Strategy: {args.mapping.upper()}")
        print(f"{'='*80}\n")

        try:
            # Generate mapping based on strategy
            if args.mapping == 'row_wise':
                path = row_wise_mapping(task_graph, grid, verbose=False)
            elif args.mapping == 'column_wise':
                path = column_wise_mapping(task_graph, grid, verbose=False)
            elif args.mapping == 'random':
                path = random_mapping(task_graph, grid, verbose=False)
            else:
                raise ValueError(f"Unknown mapping strategy: {args.mapping}")

            # Construct mapping
            mapping = {task_id: int(next_node) for task_id, _, next_node in path
                        if task_id != "start" and task_id != "end"}

            print(f"Generated {len(mapping)} task mappings using {args.mapping}")

            # Create unique filename for each run
            if args.mapping == 'random':
                config_path_file = f"{result_file}/mapping_{model_name}_{args.mapping}_{run_idx}_{spatial}_{output}_{input_split}.json"
            else:
                config_path_file = f"{result_file}/mapping_{model_name}_{args.mapping}_{spatial}_{output}_{input_split}.json"

            config_path_file_mapping = f".{config_path_file}"
            mapper = ma.Mapper()
            mapper.init(task_graph, grid)
            mapper.set_mapping(mapping)
            mapper.mapping_to_json(config_path_file_mapping,
                                    file_to_append="./config_files/arch.json")

            # Run Fast Analytical model
            print("\nRunning Fast Analytical model simulation...")
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

            # Initialize booksim results variables
            result_booksim = None
            booksim_time = None
            percentage_diff = None
            time_gain = None

            # Run Booksim2 simulation if enabled
            if args.run_booksim:
                print("\nRunning Booksim2 simulation...")
                start_time = time.time()
                result_booksim, logger = booksim_stub.run_simulation(config_path_file, dwrap=False)
                booksim_time = time.time() - start_time
                print(f"  Result: {result_booksim} cycles")
                print(f"  Time: {booksim_time:.4f} seconds")

                # Calculate metrics
                percentage_diff = abs(result_fast_anal - result_booksim) / result_booksim * 100
                time_gain = booksim_time / fast_analytical_time

                print(f"\nComparison:")
                print(f"  Difference: {abs(result_fast_anal - result_booksim)} cycles ({percentage_diff:.2f}%)")
                print(f"  Time gain: {time_gain:.4f}x")
            # Write to CSV immediately after each run
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # For random mapping, include run number in strategy name
                strategy_name = f"{args.mapping}_run{run_idx}" if args.mapping == 'random' else args.mapping

                # Build row data - always include all fields
                row_data = {
                    'mapping_strategy': strategy_name,
                    'num_partitions': total_partitions,
                    'parts_per_layer': num_partitions_per_layer[1],
                    'total_tasks': total_tasks,
                    'result_analytical': result_fast_anal,
                    'analytical_time': f"{fast_analytical_time:.4f}",
                    'partitioner_config': str(partitioner_tuples[1]),
                    'result_booksim': result_booksim if args.run_booksim else 'NaN',
                    'booksim_time': f"{booksim_time:.4f}" if args.run_booksim and booksim_time else 'NaN',
                    'percentage_diff': f"{percentage_diff:.2f}" if args.run_booksim and percentage_diff else 'NaN',
                    'time_gain': f"{time_gain:.4f}" if args.run_booksim and time_gain else 'NaN'
                }

                writer.writerow(row_data)

            print(f"\n✓ Results for {strategy_name} appended to {csv_filename}")

        except Exception as e:
            print(f"\n✗ Error with mapping strategy {args.mapping} (run {run_idx}): {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"  Skipping to next run...\n")
            continue

    print(f"\n{'='*80}")
    print(f"All runs completed!")
    print(f"Results saved to:")
    print(f"  - {csv_filename}")
    print(f"{'='*80}\n")
        









