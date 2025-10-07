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


def count_operational_layers(model):
    """Count layers that need partitioning tuples"""
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ not in ['InputLayer']:
            count += 1
    return count


if __name__ == '__main__':
    # Model setup
    #done:
    #model = AlexNet((32, 32, 3), num_classes=10, verbose=True)
    #model = VGG_16_early_layers(input_shape=(32, 32, 3), num_classes=10, verbose=True)
    #model = VGG_16_late_layers(input_shape=(4, 4, 256), num_classes=10, verbose=True)
    #model = MobileNetv1((32, 32, 3), num_classes=10, verbose=True)
    #model = ResNet32_early_blocks((32, 32, 3), verbose=True)
    
    model = ResNet32_late_blocks((16, 16, 32), num_classes=10, verbose=True)

    model = fuse_conv_bn(model, verbose=True)

    model_name = "ResNet32_late"
    result_file = "./data/mapping_comparison"

    # Prepare CSV files
    csv_filename = f"{result_file}/random_mapping_comparison_{model_name}.csv"
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

    # Configuration to test
    configuration = (3, 4, 3)  # (spatial, output, input_split)

    # Mapping strategies to test
    mapping_strategies = ['random', 10]

    # CSV header
    fieldnames = [
        'mapping_strategy',
        'num_partitions',
        'parts_per_layer',
        'total_tasks',
        'result_analytical',
        'analytical_time',
        'partitioner_config'
    ]

    # Write header first
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

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

    # Test each mapping strategy
    for i in range(0, mapping_strategies[1] + 1):


        path = random_mapping(task_graph, grid, verbose=False)

        # Construct mapping
        mapping = {task_id: int(next_node) for task_id, _, next_node in path
                    if task_id != "start" and task_id != "end"}

        print(f"Generated {len(mapping)} task mappings using {i} random strategy")

        config_path_file = f"{result_file}/mapping_{model_name}_random_{i}_{spatial}_{output}_{input_split}.json"

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

        '''
        # Run Booksim2 simulation
        print("\nRunning Booksim2 simulation...")
        start_time = time.time()
        result_booksim, logger = booksim_stub.run_simulation(config_path_file, dwrap=False)
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
                append=(strategy_idx > 1),  # Append after first strategy
                num_partitions=total_partitions,
                parts_per_layer=num_partitions_per_layer[1],
                partitioner_config=f"{mapping_strategy}_{str(partitioner_tuples[1])}"
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

        
        '''
        # Write to CSV immediately after each mapping strategy
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'mapping_strategy': 'random',
                'num_partitions': total_partitions,
                'parts_per_layer': num_partitions_per_layer[1],
                'total_tasks': total_tasks,
                'result_analytical': result_fast_anal,
                'analytical_time': f"{fast_analytical_time:.4f}",
                'partitioner_config': str(partitioner_tuples[1])
            })

        print(f"\n✓ Results for random strategy {i} appended to {csv_filename}")


    print(f"\n{'='*80}")
    print(f"All mapping strategies tested!")
    print(f"Results saved to:")
    print(f"  - {csv_filename}")
    print(f"{'='*80}\n")
        









