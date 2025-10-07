'''
==================================================
File: partitioner.py
Project: ReNOS
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati, Jakub Jastrzebski (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The partitioner.py module contains the classes and functions needed to create a partition/task graph out of a 
given DNN model. The chosen library for this purpose is TensorFlow.
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



if __name__ == '__main__':
    import csv
    import os

    #model = single_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = double_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = triple_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = ResNet_early_blocks((16, 16, 3), verbose=True)
    model = AlexNet((32, 32, 3), num_classes=10, verbose=True)
    
    #model = VGG_16_early_layers(input_shape=(32, 32, 3), num_classes=10, verbose=True)
    #model = VGG_16_late_layers(input_shape=(4, 4, 256), num_classes=10, verbose=True)

    #model = ResNet32_early_blocks((32, 32, 3), verbose=True)
    #model = ResNet32_mid_blocks((32, 32, 16), num_classes=10, verbose=True)
    #model = ResNet32_late_blocks((8, 8, 64), num_classes=10, verbose=True)
    
    #model = MobileNetv1((32, 32, 3), num_classes=10, verbose=True)
    number_of_parts = 7
    model_name = 'AlexNet'
    
    model = fuse_conv_bn(model, verbose=True)

    x_of_grid = 12
    source = 0
    drain = 143

    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

    # Generate all partition combinations
    combinations = [(i, j, k) for i in range(1, number_of_parts) for j in range(1, number_of_parts) for k in range(1, number_of_parts)]

    # Prepare CSV file
    csv_filename = f"./data/partitioner_data/partition_statistics_{model_name}.csv"
    os.makedirs("./data/partitioner_data", exist_ok=True)

    # Create CSV and write header
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'spatial', 'output', 'input_split',
            'num_partitions', 'parts_per_layer', 'partitioner_config',
            'total_tasks', 'latency_cycles', 'simulation_time_sec',
            'grid_size', 'topology'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print(f"Starting partition sweep for {model_name}...")
    print(f"Total combinations to test: {len(combinations)}")
    print(f"Results will be saved to: {csv_filename}\n")

    # Create fast analytical model stub once (reuse for all simulations)
    fast_sim = ssfam.FastAnalyticalSimulatorStub()

    best_latency = float('inf')
    best_partitioner_tuple = None
    
    # Loop over all combinations
    for idx, (i, j, k) in enumerate(combinations, 1):
        print(f"\n[{idx}/{len(combinations)}] Testing partition: spatial={i}, output={j}, input_split={k}")

        try:
            # Build partitioner tuples: first layer gets (0,1,1), rest get (i,j,k)
            partitioner_tuples = [(0, 1, 1)] + [(i, j, k)] * (len(model.layers))

            # Calculate number of partitions per layer
            num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in partitioner_tuples]
            parts_per_layer = num_partitions_per_layer[1]  # partitions for actual layers (not input)
            total_partitions = sum(num_partitions_per_layer)

            print(f"  Parts per layer: {parts_per_layer}, Total partitions: {total_partitions}")

            # Build partitions and task graph
            dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
            parts, deps = build_partitions_splitting_input_for_many_tuples(
                model, grid,
                partitioning_tuple = partitioner_tuples,
                grouping = False,
                verbose = False
            )

            task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
            total_tasks = task_graph.n_nodes

            # Generate row-wise mapping
            path = row_wise_mapping(task_graph, grid, verbose = False)
            mapping = {task_id : int(next_node) for task_id, _, next_node in path
                        if task_id != "start" and task_id != "end"}

            # Save mapping to JSON
            mapper = ma.Mapper()
            mapper.init(task_graph, grid)
            mapper.set_mapping(mapping)
            mapper.mapping_to_json(f"../data/partitioner_data/mapping_{model_name}.json",
                                    file_to_append="./config_files/arch.json")

            # Run fast analytical simulation
            start_time = time.time()
            result_fast_anal, logger_fast_anal = fast_sim.run_simulation(
                f"./data/partitioner_data/mapping_{model_name}.json",
                verbose=False
            )
            simulation_time = time.time() - start_time

            print(f"  Latency: {result_fast_anal} cycles, Sim time: {simulation_time:.4f}s")

            # Write results to CSV
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'partitioner_config': str(partitioner_tuples[1]),
                    'spatial': i,
                    'output': j,
                    'input_split': k,
                    'num_partitions': total_partitions,
                    'parts_per_layer': parts_per_layer,
                    'total_tasks': total_tasks,
                    'simulation_time_sec': simulation_time,
                    'grid_size': f"{x_of_grid}x{x_of_grid}",
                    'topology': grid.topology.name,
                    'latency_cycles': result_fast_anal
                })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            # Log failed combinations
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'spatial': i,
                    'output': j,
                    'input_split': k,
                    'num_partitions': 'ERROR',
                    'parts_per_layer': 'ERROR',
                    'partitioner_config': str((i, j, k)),
                    'total_tasks': 'ERROR',
                    'latency_cycles': 'ERROR',
                    'simulation_time_sec': 'ERROR',
                    'grid_size': f"{x_of_grid}x{x_of_grid}",
                    'topology': grid.topology.name
                })
            continue
        
        # Track best configuration
        if result_fast_anal < best_latency:
            best_latency = result_fast_anal
            best_partitioner_tuple = (i, j, k)
            print(f"  New best latency found: {best_latency} cycles with partitioner {best_partitioner_tuple}")
            

    print(f"\n{'='*80}")
    print(f"Partition sweep complete!")
    print(f"Results saved to: {csv_filename}")
    print(f"Total combinations tested: {len(combinations)}")
    if best_partitioner_tuple:
        print(f"Best latency: {best_latency} cycles with partitioner (spatial={best_partitioner_tuple[0]}, output={best_partitioner_tuple[1]}, input_split={best_partitioner_tuple[2]})")
    else:
        print("No valid partitioner configuration found.")
    print(f"{'='*80}")

    # Booksim2 simulation (can be uncommented for comparison)
    # print("Booksim2 simulation...")
    # stub = ss.SimulatorStub()
    # start_time = time.time()
    # result, logger = stub.run_simulation("./data/partitioner_data/mapping.json", dwrap=True)
    # booksim_time = time.time() - start_time
    # print(f"Booksim2 result: {result}")
    # print(f"Booksim2 simulation time: {booksim_time:.4f} seconds")
    #
    # percentage_diff_fast_anal = abs(result_fast_anal - result) / result * 100
    # print("\nComparison of results:")
    # print(f"Difference fast analytical: {abs(result_fast_anal - result)} cycles ({percentage_diff_fast_anal:.2f}%)")
    # print(f"Booksim2 simulation time: {booksim_time:.4f} seconds")
    # print(f"Fast Analytical model simulation time: {fast_analytical_time:.4f} seconds")
    # print(f"Time gain fast analytical: {booksim_time / fast_analytical_time:.4f}x")

    print("Partitioner Done!")









