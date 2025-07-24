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
from visualizer import plot_timeline


model = single_conv((10, 10, 4), num_classes=1, verbose=True)
conv_layer = model.layers[1]  # Get the first conv layer

x_of_grid = 4
source = 0
drain = 15

grid = dm.Grid()
grid.init(x_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

# Explore all partitioning combinations
spatial, output, input_split = search_space_split_factors(
    conv_layer, 
    factor=2,  # Max splitting factor
    FLOP_threshold=3e6,
    size_of_grid = x_of_grid**2,
    return_best_valid=True,
    path = "data/partitioner_data"
)

print(f"Best partitioning factors found: spatial={spatial}, output={output}, input_split={input_split}")

partitioner_tuple = (spatial, output, input_split)

#### Model analysis and partitioning ####

print("")
print("Analysis of the model...")
analyze_ops(model, incl_info = True)
    
dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
#spatial, output, input
parts, deps = build_partitions_splitting_input(model, grid, partitioning_tuple = partitioner_tuple, grouping = False, verbose = True)

#print partitions and dependencies in a table format
print("")
print("Analysis of the partitions...") 
print_partitions_table_adaptive(parts, deps, mode="auto") #possible modes: "auto", "compact", "vertical", "minimal"

print("Plotting the partitions and dependencies of the model...")
plot_partitions(parts, deps, namefile = 'data/partitioner_data/task_graph.png')
print("Done!")

task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
path = choose_path_simply(task_graph, grid, ass_factor = 16, verbose = True)

# constuct the mapping form the path
mapping = {task_id : int(next_node) for task_id, _, next_node in path if task_id != "start" and task_id != "end"}

print(f"Mapping: {mapping}")

mapper = ma.Mapper()
mapper.init(task_graph, grid)
mapper.set_mapping(mapping)
mapper.mapping_to_json("../data/partitioner_data/mapping.json", file_to_append="./config_files/arch.json")

#stub = ss.SimulatorStub()
#result, logger = stub.run_simulation("../data/partitioner_data/mapping.json", dwrap=True)

plot_timeline("./data/partitioner_data/mapping.json", timeline_path = "./data/partitioner_data/timeline.png", verbose = True)

print("Partitioner Done!")









