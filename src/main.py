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
import visualizer
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model

from models import *


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
        
    print(f"Selected algorithm: {args.algo}")
        
    # Initialize global directories
    initialize_globals(args.algo)
    # Get the shared timestamp
    # Debug - Print the global variables to check they're set correctly
    print(f"After initialization:")
    debug_globals()
    
    #measute time of the optmiization
    start = time.time()
    model = test_model((28, 28, 1), verbose= True)
    # model = small_test_model((28, 28, 1))
    # model = load_model("ResNet50")
    # model = load_model("MobileNet")
    # model = load_model("MobileNetV2")

    # # # plot_model(model, to_file="visual/model.png", show_shapes=True)
    # # # analyze_ops(model, True)
    
    # grid is: number of processor x number of processors (size_of_grid x size_of_grid)
    size_of_grid = 6
    source = 0
    drain = 35
    
    assert drain < size_of_grid * size_of_grid, "Drain point cannot exceed size_of_grid x size_of_grid - 1"
    
    if ACO:
        print("Running Ant Colony Optimization...")
        # Redirect stdout to the Logger
        log_path = get_aco_log_path()
        sys.stdout = Logger(log_path)
        #drain point cannot exceed size_of_grid x size_of_grid - 1
        task_graph = model_to_graph(model, source = source, drain = drain, verbose=False)
        #plot_graph(task_graph)
        #print_dependencies(task_graph)

        grid = dm.Grid()
        grid.init(size_of_grid, 2, dm.Topology.TORUS)

        params = op.ACOParameters(
            n_ants = 10,
            rho = 0.05,
            n_best = 10,
            n_iterations = 150,
            alpha = 1.,
            beta = 1.2,
        )
        n_procs = 10
        #opt = op.AntColony( params, grid, task_graph, seed = None)
        opt = op.ParallelAntColony(n_procs, params, grid, task_graph, seed = 44)
        
        shortest = opt.run(once_every=1, show_traces= False)
        print("The best path found is: ")
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
        # Redirect stdout to the Logger
        log_path = get_ga_log_path()
        sys.stdout = Logger(log_path)
        
        #drain point cannot exceed size_of_grid x size_of_grid - 1
        task_graph = model_to_graph(model, source = source, drain = drain, verbose=False)
        #plot_graph(task_graph)

        grid = dm.Grid()
        grid.init(size_of_grid, 2, dm.Topology.TORUS)
        
        params = op.GAParameters(
        sol_per_pop = 30,
        n_parents_mating=20,
        keep_parents= 10,
        parent_selection_type= "sss",
        n_generations = 2, #800,
        mutation_probability = .7,
        crossover_probability = .7,
        )
        
        n_procs = 50
        opt = op.GeneticAlgorithm(params, grid, task_graph, seed = None)
        #opt = op.ParallelGA(n_procs, params, grid, task_graph, seed = None)
                
        shortest = opt.run()
        # #opt.ga_instance.plot_fitness()
        print("The best path found is: ")
        print(shortest[0], 1/shortest[1])
        
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n ... End Genetic Algoriothm Optimization")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("Done!")



