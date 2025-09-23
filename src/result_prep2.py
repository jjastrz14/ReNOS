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
from latency_energy_analysis import analyze_logger_events, plot_parallel_execution_comparison
from latency_energy_analysis import parallel_analysis_to_dataframe
import json
import time
import os
import math


def generate_split_factors(num_partitions, strategy):
    """
    Generate split factor tuples for all layers based on number of partitions and strategy.

    Parameters:
    -----------
    num_partitions : int
        Desired number of partitions (must be power of 2)
    strategy : str
        Partitioning strategy: 'spatial', 'output', or 'input'

    Returns:
    --------
    dict with split factors for each layer
    """

    # Calculate the exponent (log base 2 of num_partitions)
    if num_partitions & (num_partitions - 1) != 0:
        raise ValueError("num_partitions must be a power of 2")

    exp = int(math.log2(num_partitions))

    # Define split patterns for each strategy
    if strategy == "spatial":
        # Spatial partitioning: split along spatial dimension (first dimension)
        conv_factors = (exp, 1, 1)
        dense_factors = (exp, 1, 1)  # Fully connected layers split along output
    elif strategy == "output":
        # Output partitioning: split along output channels (second dimension)
        conv_factors = (0, num_partitions, 1)
        dense_factors = (exp, 1, 1)  # Fully connected layers split along output
    elif strategy == "input":
        # Input partitioning: split along input channels (third dimension)
        conv_factors = (0, 1, num_partitions)
        dense_factors = (exp, 1, 1)  # Fully connected layers split along output
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply same partitioning to all layers (except input which stays unpartitioned)
    #early layers resnet
    return {
        'input': (0, 1, 1),        # Input layer typically not partitioned
        'conv2d': conv_factors,
        'bn': conv_factors,
        'relu': conv_factors,
        'conv2d_1': conv_factors,
        'bn_1': conv_factors,
        'conv2d_2': conv_factors,
        'bn_2': conv_factors,
    }
    
    #late layers resnet
    #return {
    #    'input': (0, 1, 1),        # Input layer typically not partitioned
    #    'conv2d': conv_factors,
    #    'conv2d_3': conv_factors,
    #    'conv2d_6': conv_factors,
    #    'batch_normalization': conv_factors,
    #    'batch_normalization_2': conv_factors,
    #    'batch_normalization_4': conv_factors,
    #    'conv2d_1': conv_factors,
    #    'conv2d_4': conv_factors,
    #    'conv2d_7': conv_factors,
    #    'batch_normalization_1': conv_factors,
    #    'batch_normalization_3': conv_factors,
    #   'batch_normalization_5': conv_factors,
    #    'output1': dense_factors,
    #    'output2': dense_factors,
    #    'output3': dense_factors,
    #}

    


if __name__ == '__main__':
    #model = single_conv_big((32, 32, 256), num_classes=1, verbose=True)
    #model = single_conv((10, 10, 3), num_classes=1, verbose=False)
    #model = ResNet_early_block_test((32, 32, 3), verbose=True)
    model = ResNet_early_blocks((32,32,3), verbose=True)
    #model = ResNet_late_blocks(input_shape=(8, 8, 256), verbose=True)

    all_results = []
        
    date = "22Sept_data"
    path = "./data/22Sept_data"
    model_name = "model_resnet_early_blocks"
    
    x_of_grid = 12
    source = 0
    drain = 143
    grid = dm.Grid()
    topology = dm.Topology.TORUS
    grid.init(x_of_grid, 2, topology, source = source, drain = drain)
    
    if topology == dm.Topology.MESH:
        topology_name = "mesh"
    elif topology == dm.Topology.TORUS: 
        topology_name = "torus"
    else:
        raise ValueError("Unsupported topology")
    
    for i in range(10,51):
        strategy=f"mixed_random_{i}"
        
        partitioner_tuples=[(0,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1), (4,1,1)]
        num_partitions = 2**partitioner_tuples[1][0]*partitioner_tuples[1][1]*partitioner_tuples[1][2]
        
        dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
        #spatial, output, input
        parts, deps = build_partitions_splitting_input_for_many__tuples(model, grid, partitioning_tuple = partitioner_tuples, grouping = False, verbose = True)
        
        #instead of optmisation step, just map following a simple path from source to drain
        task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
        assignment_pes = random_mapping(task_graph, grid, verbose = False)
        # constuct the mapping form the path
        mapping = {task_id : int(next_node) for task_id, _, next_node in assignment_pes if task_id != "start" and task_id != "end"}

        mapper = ma.Mapper()
        mapper.init(task_graph, grid)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(f".{path}/mapping_{strategy}.json", file_to_append="./config_files/arch.json")

        with open(f"{path}/mapping_{strategy}.json", "r") as f:
            data = json.load(f)
            data["arch"]["logger"] = 1
        with open(f"{path}/mapping_{strategy}.json", "w") as f:
            json.dump(data, f, indent = 4)
        
        # Measure Booksim2 simulation time
        print("Booksim2 simulation...")
        stub = ss.SimulatorStub()
        start_time = time.time()
        result, logger = stub.run_simulation(f"{path}/mapping_{strategy}.json", dwrap = False, verbose = False)
        booksim_time = time.time() - start_time
        print(f"Booksim2 result: {result}")
        print(f"Booksim2 simulation time: {booksim_time} seconds")
        
        df1, df2, parallel = analyze_logger_events(logger)
        
        #save parallel to csv:
        parallel_df = parallel_analysis_to_dataframe(parallel, strategy_name=strategy)
        #add column between column 0 and 1 with number of partitions
        parallel_df.insert(1, 'partitions', num_partitions)
        #save to csv
        parallel_df.to_csv(f"{path}/{model_name}_{x_of_grid}x{x_of_grid}_{topology_name}_{strategy}.csv", index=False)