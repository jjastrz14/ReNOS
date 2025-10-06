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
    model = ResNet32_early_blocks((32, 32, 3), verbose=True)
    model = fuse_conv_bn(model, verbose=True)
    
    model_name = "ResNet32_early_blocks"
    result_file = "./data/light_noc_comp"
    
    # Prepare CSV files
    csv_filename = f"{result_file}/p3_noc_comparison_results_{model_name}.csv"
    energy_csv_filename = f"{result_file}/p3_latency_energy_results_{model_name}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Grid setup
    x_of_grid = 8
    source = 0
    drain = 63
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
        'partitioner_config'
    ]

    # Write header first
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Generate all partition configurations incrementally
    # Pattern: (1,1,1), (2,1,1), (2,2,1), (2,2,2), (3,2,2), (3,3,2), (3,3,3), etc.
    #configurations = []
    #for spatial in range(1, 3):
    #    for output in range(1, spatial + 1):
    #        for input_split in range(1, output + 1):
    #            configurations.append((spatial, output, input_split))

    
    #another option is to manually define a few configurations
    #configuration p1
    #configurations= [
    #(1, 1, 1), (1, 1, 2), (1, 1, 3),
    #(1, 2, 1), (1, 2, 2), (1, 2, 3),
    #(1, 3, 1), (1, 3, 2), (1, 3, 3),
    #(2, 1, 1), (2, 1, 2), (2, 1, 3),
    #(2, 2, 1), (2, 2, 2), (2, 2, 3),
    #(2, 3, 1), (2, 3, 2), (2, 3, 3),
    #(3, 1, 1), (3, 1, 2), (3, 1, 3),
    #(3, 2, 1), (3, 2, 2), (3, 2, 3),
    #(3, 3, 1), (3, 3, 2), (3, 3, 3)
    #]
    #configuration p2
    #configurations = [ (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 2, 4), (1, 2, 5), (1, 2, 6),(1, 3, 4),
    #                    (1, 3, 5), (1, 3, 6),(1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 4, 4), (1, 4, 5),
    #                    (1, 4, 6), (1, 5, 1), (1, 5, 2), (1, 5, 3), (1, 5, 4), (1, 5, 5), (1, 5, 6),
    #                    (1, 6, 1), (1, 6, 2), (1, 6, 3), (1, 6, 4), (1, 6, 5), (1, 6, 6)]
    #configuration p3
    configurations = [  (2, 1, 4),
                        (2, 1, 5),
                        (2, 1, 6),
                        (2, 2, 4),
                        (2, 2, 5),
                        (2, 2, 6),
                        (2, 3, 4),
                        (2, 3, 5),
                        (2, 3, 6),
                        (2, 4, 1),
                        (2, 4, 2),
                        (2, 4, 3),
                        (2, 4, 4),
                        (2, 4, 5),
                        (2, 4, 6),
                        (2, 5, 1),
                        (2, 5, 2),
                        (2, 5, 3),
                        (2, 5, 4),
                        (2, 5, 5),
                        (2, 5, 6),
                        (2, 6, 1),
                        (2, 6, 2),
                        (2, 6, 3),
                        (2, 6, 4),
                        (2, 6, 5),
                        (2, 6, 6)]

    print(f"Generated {len(configurations)} configurations to test\n")
    # Iterate over configurations    
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
            
            config_path_file = result_file + f"/mapping_{spatial}_{output}_{input_split}.json"
            
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
                    partitioner_config = str(partitioner_tuples[1])
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
                    'partitioner_config': str(partitioner_tuples[1])
                })

            print(f"✓ Results appended to {csv_filename}")

        except Exception as e:
            print(f"\n✗ Error with config {config}: {str(e)}")
            print(f"  Skipping to next iteration...\n")
            continue

    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_filename}")
    print(f"{'='*80}\n")
        









