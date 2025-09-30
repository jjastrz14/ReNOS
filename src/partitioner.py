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
import time



if __name__ == '__main__':
    #model = single_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = double_conv((10, 10, 4), num_classes=1, verbose=True)
    #model = triple_conv((10, 10, 4), num_classes=1, verbose=True)
    model = ResNet_early_blocks((16, 16, 3), verbose=True)
    #model = LeNet4((28, 28, 1), num_classes=10, verbose=True)
    
    x_of_grid = 4
    source = 0
    drain = 15

    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

    #partitioner_tuples = []
    #for layer in model.layers:
    #    # Explore all partitioning combinations
    #    spatial, output, input_split = search_space_split_factors(
    #        layer, 
    #        factor=4,  # Max splitting factor
    #        FLOP_threshold=3e6,
    #        size_of_grid = x_of_grid**2,
    #        return_best_valid=True,
    #        path = "data/partitioner_data"
    #    )
    #    print(f"Layer {layer.name}: spatial={spatial}, output={output}, input_split={input_split}")
    #    partitioner_tuples.append((spatial, output, input_split))

    #partitioner_tuples = [(0, 1, 1), (3,2,2),(3,2,2)]
    
    partitioner_tuples = [(0, 1, 1), (1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,4,4), (1,4,4)]
    #for lenet: 
    #[(0, 1, 1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1)]]
    #print(f"Best partitioning factors found: spatial={spatial}, output={output}, input_split={input_split}")

    #partitioner_tuple = (spatial, output, input_split)

    #### Model analysis and partitioning ####
    num_partitions_per_layer = [2**x[0] * x[1] * x[2] for x in partitioner_tuples]
    print(f"number of partitions per layer: {num_partitions_per_layer}")
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
    path = row_wise_mapping(task_graph, grid, verbose = False)

    # constuct the mapping form the path
    mapping = {task_id : int(next_node) for task_id, _, next_node in path if task_id != "start" and task_id != "end"}

    #print(f"Mapping: {mapping}")
    mapper = ma.Mapper()
    mapper.init(task_graph, grid)
    mapper.set_mapping(mapping)
    mapper.mapping_to_json("../data/partitioner_data/mapping.json", file_to_append="./config_files/arch.json")
    
    print("Fast Analytical model simulation...")
    # Create the stub
    fast_sim = ssfam.FastAnalyticalSimulatorStub()
    # Run simulation
    start_time = time.time()
    result_fast_anal, logger_fast_anal = fast_sim.run_simulation("./data/partitioner_data/mapping.json", verbose=True)
    fast_analytical_time = time.time() - start_time
    print(f"Fast Analytical model result: {result_fast_anal}")
    print(f"Fast Analytical model simulation time: {fast_analytical_time:.4f} seconds")

    # Measure Booksim2 simulation time
    print("Booksim2 simulation...")
    stub = ss.SimulatorStub()
    start_time = time.time()
    result, logger = stub.run_simulation("./data/partitioner_data/mapping.json", dwrap=True)
    booksim_time = time.time() - start_time
    print(f"Booksim2 result: {result}")
    print(f"Booksim2 simulation time: {booksim_time:.4f} seconds")

    
    percentage_diff_fast_anal = abs(result_fast_anal - result) / result * 100
    print("\nComparison of results:")
    print(f"Difference fast analytical: {abs(result_fast_anal - result)} cycles ({percentage_diff_fast_anal:.2f}%)")
    print(f"Booksim2 simulation time: {booksim_time:.4f} seconds")
    print(f"Fast Analytical model simulation time: {fast_analytical_time:.4f} seconds")
    
    # Compare simulation times
    print(f"Time gain fast analytical: {booksim_time / fast_analytical_time:.4f}x")
    

    visualise = True
    if visualise:
        plot_timeline("./data/partitioner_data/mapping.json", timeline_path = "./data/partitioner_data/timeline.png", verbose = False)
        # Visualize analytical model simulation
        #total_latency, visualizer = visualize_simulation(stub_anal.fast_sim, "./data/partitioner_data/mapping.json", timeline_path="./data/partitioner_data/timeline_analytical.png", utilization_path="./data/partitioner_data/utilization.png")
        
    print("Partitioner Done!")









