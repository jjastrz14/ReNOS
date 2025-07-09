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

from graph import model_to_graph
import domain as dm
from utils.partitioner_utils import search_space_split_factors
from graph import TaskGraph
from visualizer import plot_timeline
from models import *


model = single_conv((28, 28, 1), num_classes=10, verbose=True)
conv_layer = model.layers[1]  # Get the first conv layer

# Explore all partitioning combinations
spatial, output, input_split = search_space_split_factors(
    conv_layer, 
    factor=10,  # Max splitting factor
    FLOP_threshold=3e6,
    return_best_valid=True,
    path = "data/partitioner_data"
)

print("Partitioner Done!")









