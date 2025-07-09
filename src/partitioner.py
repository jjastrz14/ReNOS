'''
==================================================
File: partitioner.py
Project: simopty
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The partitioner.py module contains the classes and functions needed to create a partition/task graph out of a 
given DNN model. The chosen library for this purpose is TensorFlow.
"""

import os
import time
import argparse
import graph as dg
from graph import model_to_graph
import domain as dm
import mapper as mp
import simulator_stub as ss
from dirs import *
import optimizers as op
from utils.plotting_utils import *
from utils.ga_utils import *
from utils.partitioner_utils import *
from utils.ani_utils import *
from graph import TaskGraph
from visualizer import plot_timeline
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
from models import *

model = LeNet4((28, 28, 1), verbose=True)
# model = Resnet9s((32, 32, 3), verbose=True)
# model = test_conv((28, 28, 1), num_classes = 2, verbose=True)
# model = test_model((28, 28, 1), verbose= True)
# model = small_test_model((28, 28, 1))
# model = load_model("ResNet50")
# model = load_model("MobileNet")
# model = load_model("MobileNetV2")

# grid is: number of processor x number of processors (size_of_grid x size_of_grid)
size_of_grid = 4
source = 0
drain = 15

grid = dm.Grid()
grid.init(size_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)

#### Model analysis and partitioning ####

print("")
print("Analysis of the model...")
analyze_ops(model, incl_info = True)
    
dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
parts, deps = build_partitions(model, grid, grouping = False, verbose = True)
    
#print partitions and dependencies in a table format
print("")
print("Analysis of the partitions...")
print_partitions_table_adaptive(parts, deps, mode="auto")
        
#print("Plotting the partitions and dependencies of the model...")
#plot_partitions(parts, deps)
#print("Done!")
    
task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)

print("Partitioner Done!")









