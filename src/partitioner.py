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
import analytical_simulator_stub as ssam
from visualizer import plot_timeline
from utils.ani_utils import visualize_simulation
import time


if __name__ == '__main__':
    model = single_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = double_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = triple_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = ResNet_early_blocks((16, 16, 3), verbose=True)
    #model = LeNet4((28, 28, 1), num_classes=10, verbose=True)
    
    x_of_grid = 4
    source = 0
    drain = 15

    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

    partitioner_tuples = []
    for layer in model.layers:
        # Explore all partitioning combinations
        spatial, output, input_split = search_space_split_factors(
            layer, 
            factor=4,  # Max splitting factor
            FLOP_threshold=3e6,
            size_of_grid = x_of_grid**2,
            return_best_valid=True,
            path = "data/partitioner_data"
        )
        print(f"Layer {layer.name}: spatial={spatial}, output={output}, input_split={input_split}")
        partitioner_tuples.append((spatial, output, input_split))

    #print(f"Best partitioning factors found: spatial={spatial}, output={output}, input_split={input_split}")

    #partitioner_tuple = (spatial, output, input_split)

    #### Model analysis and partitioning ####

    print("")
    #print("Analysis of the model...")
    #analyze_ops(model, incl_info = True)
        
    dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
    #spatial, output, input
    parts, deps = build_partitions_splitting_input_for_many__tuples(model, grid, partitioning_tuple = partitioner_tuples, grouping = False, verbose = True)

    #print partitions and dependencies in a table format
    #print("")
    #print("Analysis of the partitions...") 
    #print_partitions_table_adaptive(parts, deps, mode="auto") #possible modes: "auto", "compact", "vertical", "minimal"

    #print("Plotting the partitions and dependencies of the model...")
    #plot_partitions(parts, deps, namefile = 'data/partitioner_data/task_graph.png')
    #print("Done!")

    #instead of optmisation step, just map following a simple path from source to drain
    task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
    path = choose_path_simply(task_graph, grid, ass_factor = x_of_grid**2, verbose = False)

    # constuct the mapping form the path
    mapping = {task_id : int(next_node) for task_id, _, next_node in path if task_id != "start" and task_id != "end"}

    #print(f"Mapping: {mapping}")
    mapper = ma.Mapper()
    mapper.init(task_graph, grid)
    mapper.set_mapping(mapping)
    mapper.mapping_to_json("../data/partitioner_data/mapping.json", file_to_append="./config_files/arch.json")

    # Measure Booksim2 simulation time
    print("Booksim2 simulation...")
    stub = ss.SimulatorStub()
    start_time = time.time()
    result, logger = stub.run_simulation("./data/partitioner_data/mapping.json", dwrap=True)
    booksim_time = time.time() - start_time
    print(f"Booksim2 result: {result}")

    # Measure Analytical Model simulation time
    print("Analytical model simulation...")
    stub_anal = ssam.AnalyticalSimulatorStub()
    start_time = time.time()
    result_anal, logger_anal = stub_anal.run_simulation("./data/partitioner_data/mapping.json", dwrap=True)
    analytical_time = time.time() - start_time
    print(f"Analytical model result: {result_anal}")
    
    percentage_diff = abs(result_anal - result) / result * 100
    ratio = result_anal / result
    print(f"\nDifference: {abs(result_anal - result)} cycles ({percentage_diff:.2f}%)")
    #print(f"Analytical/BookSim2 ratio: {ratio:.2f}x")
    print(f"Analytical model simulation time: {analytical_time:.4f} seconds")
    print(f"Booksim2 simulation time: {booksim_time:.4f} seconds")
    # Compare simulation times
    time_ratio = booksim_time / analytical_time
    print(f"Time gain: {time_ratio:.4f}x")

    visualise = False
    if visualise:
        plot_timeline("./data/partitioner_data/mapping.json", timeline_path = "./data/partitioner_data/timeline.png", verbose = False)
        # Visualize analytical model simulation
        total_latency, visualizer = visualize_simulation(stub_anal.fast_sim, "./data/partitioner_data/mapping.json", timeline_path="./data/partitioner_data/timeline_analytical.png", utilization_path="./data/partitioner_data/utilization.png")
        
    print("Partitioner Done!")









