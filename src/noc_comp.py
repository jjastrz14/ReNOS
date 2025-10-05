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
import time
import csv
import os


def count_operational_layers(model):
    """Count layers that need partitioning tuples"""
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ not in ['InputLayer']:
            count += 1
    return count


if __name__ == '__main__':
    # Model setup
    model = ResNet32_early_blocks((32, 32, 3), verbose=True)
    model = fuse_conv_bn(model, verbose=True)

    # Grid setup
    x_of_grid = 8
    source = 0
    drain = 63
    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source=source, drain=drain)

    # Get number of layers that need tuples
    num_layers = count_operational_layers(model)
    print(f"\nModel has {num_layers} operational layers requiring tuples\n")

    # Prepare CSV file
    csv_filename = "./data/light_noc_comp/noc_comparison_results.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

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
        'partitioner_config'
    ]

    # Write header first
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Generate all partition configurations incrementally
    # Pattern: (1,1,1), (2,1,1), (2,2,1), (2,2,2), (3,2,2), (3,3,2), (3,3,3), etc.
    configurations = []
    for spatial in range(1, 5):
        for output in range(1, spatial + 1):
            for input_split in range(1, output + 1):
                configurations.append((spatial, output, input_split))

    print(f"Generated {len(configurations)} configurations to test\n")

    iteration = 0
    for config in configurations:
        iteration += 1
        spatial, output, input_split = config
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{len(configurations)}: Config (spatial={spatial}, output={output}, input={input_split})")
        print(f"{'='*80}\n")

        # Build partitioner tuples: first tuple is (0,1,1), rest are (spatial, output, input_split)
        partitioner_tuples = [(0, 1, 1)] + [(spatial, output, input_split)] * (num_layers)

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

            task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
            path = row_wise_mapping(task_graph, grid, verbose=False)

            # Construct mapping
            mapping = {task_id: int(next_node) for task_id, _, next_node in path
                      if task_id != "start" and task_id != "end"}

            mapper = ma.Mapper()
            mapper.init(task_graph, grid)
            mapper.set_mapping(mapping)
            mapper.mapping_to_json("../data/light_noc_comp/mapping.json",
                                  file_to_append="./config_files/arch.json")

            # Run Fast Analytical model
            print("Running Fast Analytical model simulation...")
            fast_sim = ssfam.FastAnalyticalSimulatorStub()
            start_time = time.time()
            result_fast_anal, logger_fast_anal = fast_sim.run_simulation(
                "./data/light_noc_comp/mapping.json", verbose=False
            )
            fast_analytical_time = time.time() - start_time
            print(f"  Result: {result_fast_anal} cycles")
            print(f"  Time: {fast_analytical_time:.4f} seconds")

            # Run Booksim2 simulation
            print("\nRunning Booksim2 simulation...")
            stub = ss.SimulatorStub()
            start_time = time.time()
            result_booksim, logger = stub.run_simulation(
                "./data/light_noc_comp/mapping.json", dwrap=True
            )
            booksim_time = time.time() - start_time
            print(f"  Result: {result_booksim} cycles")
            print(f"  Time: {booksim_time:.4f} seconds")

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
                    'partitioner_config': str(partitioner_tuples[1])
                })

            print(f"\n✓ Config {config} completed successfully")
            print(f"✓ Results appended to {csv_filename}")

        except Exception as e:
            print(f"\n✗ Error with config {config}: {str(e)}")
            print(f"  Skipping to next iteration...\n")
            continue

    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_filename}")
    print(f"{'='*80}\n")
        









