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
import os


if __name__ == '__main__':
    #model = single_conv_big((32, 32, 256), num_classes=1, verbose=True)
    model = single_conv((10, 10, 3), num_classes=1, verbose=False)

    date = "18Sept_data"
    path = "./data/18Sept_data"

    partitioner_tuples = []
    for layer in model.layers:
        # Explore all partitioning combinations
        spatial, output, input_split = split_factor_only_one_strategy(
            layer, 
            strategy = "spatial", # "spatial", "output", "input"
            factor=8,  # Max splitting factor
            path = path
        )
        print(f"Layer {layer.name}: spatial={spatial}, output={output}, input_split={input_split}")
        partitioner_tuples.append((spatial, output, input_split))
    
    strategy = "spatial"  # "spatial", "output", "input"
    x_of_grid = 3
    source = 0
    drain = 8
    grid = dm.Grid()
    grid.init(x_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

    partitioner_tuples = [(0,1,1), (1,1,1)] #input, conv2d
    dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
    #spatial, output, input
    parts, deps = build_partitions_splitting_input_for_many__tuples(model, grid, partitioning_tuple = partitioner_tuples, grouping = False, verbose = True)

    #instead of optmisation step, just map following a simple path from source to drain
    task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
    path = choose_path_simply(task_graph, grid, ass_factor = x_of_grid**2, verbose = False)
    # constuct the mapping form the path
    mapping = {task_id : int(next_node) for task_id, _, next_node in path if task_id != "start" and task_id != "end"}

    #print(f"Mapping: {mapping}")
    mapper = ma.Mapper()
    mapper.init(task_graph, grid)
    mapper.set_mapping(mapping)
    mapper.mapping_to_json(f"../data/18Sept_data/mapping_{strategy}.json", file_to_append="./config_files/arch.json")
    
    # Measure Booksim2 simulation time
    print("Booksim2 simulation...")
    stub = ss.SimulatorStub()
    start_time = time.time()
    result, logger = stub.run_simulation(f"./data/18Sept_data/mapping_{strategy}.json", dwrap=True)
    booksim_time = time.time() - start_time
    print(f"Booksim2 result: {result}")
    print(f"Booksim2 simulation time: {booksim_time} seconds")