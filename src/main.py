'''
==================================================
File: main.py
Project: simopty
File Created: Sunday, 8th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The main.py module contains the main function of the program.
"""

import os
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
import visualizer
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import time
import argparse

def test_model(input_shape):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='linear') (inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, kernel_size=(5, 5))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    summary = model.summary()
    return model


def conv_layer(input_shape):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='linear') (inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    summary = model.summary()
    return model


def load_model(model_name):
    available_models = ["ResNet50", "MobileNetV2", "MobileNet", "ResNet18"]
    if model_name not in available_models:
        raise ValueError(f"Model not available. Please choose from the following: {', '.join(available_models)}")
    
    # Load the model
    model = keras.applications.__getattribute__(model_name)(weights='imagenet')
    return model

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select optimization algorithm.")
    parser.add_argument("-algo", choices=["ACO", "GA"], required=True, help="Choose 'ACO' for Ant Colony Optimization or 'GA' for Genetic Algorithm.")
    args = parser.parse_args()

    # Set flags based on the argument
    if args.algo == "ACO":
        ACO = True
        GA = False
    elif args.algo == "GA":
        ACO = False
        GA = True
    
    start = time.time()
    
    #model = test_model((28, 28, 1))
    model = conv_layer((28, 28, 1))
    # # # model = load_model("ResNet50")
    # # # model = load_model("MobileNet")
    # # # model = load_model("MobileNetV2")

    # # # model.summary()
    # # # plot_model(model, to_file="visual/model.png", show_shapes=True)
    # # # analyze_ops(model, True)

    task_graph = model_to_graph(model, verbose=True)
    plot_graph(task_graph)

    grid = dm.Grid()
    grid.init(5, 2, dm.Topology.TORUS)
    
    if ACO:
        print("Running Ant Colony Optimization...")

        params = op.ACOParameters(
            n_ants = 100,
            rho = 0.05,
            n_best = 10,
            n_iterations = 1,
            alpha = 1.,
            beta = 1.2,
        )
        n_procs = 50
        #opt = op.AntColony( params, grid, task_graph, seed = None)
        opt = op.ParallelAntColony(n_procs, params, grid, task_graph, seed = 2137)
        
        print("Path to used arch.json file in ACO: ", ARCH_FILE)

        shortest = opt.run(once_every=1, show_traces= False)
        print(shortest)
        
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n ... End Ant Colony Optimization")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("Done!")

    if GA: 
        
        print("Running Genetic Algoriothm Optimization...")
        
        params = op.GAParameters(
        sol_per_pop =100,
        n_parents_mating=20,
        keep_parents= 10,
        parent_selection_type= "sss",
        n_generations = 1,#800,
        mutation_probability = .7,
        crossover_probability = .7,
        )
        
        n_procs = 50
        #opt = op.GeneticAlgorithm(params, grid, task_graph, seed = None)
        opt = op.ParallelGA(n_procs, params, grid, task_graph, seed = 2137)
        
        print("Path to used arch.json file in GA: ", ARCH_FILE)
        
        shortest = opt.run()
        # #opt.ga_instance.plot_fitness()
        print(shortest[0], 1/shortest[1])
        
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n ... End Genetic Algoriothm Optimization")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("Done!")
        
    # # file_name_json = "/../runs/best_solution.json"
    # # path_timeline = "visual/timeline.png"
    # file_name_json = "/../../data/ACO/best_solution.json"
    # path_timeline = "data/ACO/timeline.png"
    # visualizer.plot_timeline(file_name_json, path_timeline, verbose = False)
    # print(opt.path_length(shortest[0], verbose =False))
    # # # Load the statistics and plot the results
    # # stats = np.load("data/statistics.npy", allow_pickle=True).item()
    # # print(stats)



