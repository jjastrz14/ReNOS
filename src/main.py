'''
==================================================
File: main.py
Project: simopty
File Created: Sunday, 8th December 2024
Authors: Jakub Jastrzebski and Edoardo Cabiati (jakubandrzej.jastrzebski@polimi.it, edoardo.cabiati@mail.polimi.it)
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
from graph import TaskGraph
from visualizer import plot_timeline
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model

from models import *


if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select optimization algorithm and name of the resulting directory.")
    parser.add_argument("-algo", choices=["ACO", "GA"], required=True, help="Choose 'ACO' for Ant Colony Optimization or 'GA' for Genetic Algorithm.")
    parser.add_argument("-name", type=str, default="_run_", required=True, help="Choose name of the resulting directory (default: '_run_').")
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
    initialize_globals(args.algo, args.name)
    # Get the shared timestamp
    # Debug - Print the global variables to check they're set correctly
    print(f"After initialization:")
    debug_globals()
    
    #measute time of the optmiization
    start = time.time()
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
    #spatial, output, input
    parts, deps = build_partitions(model, grid, chosen_splitting_strategy = "input", grouping = False, verbose = True)
    
    #print partitions and dependencies in a table format
    print("")
    print("Analysis of the partitions...") 
    print_partitions_table_adaptive(parts, deps, mode="minimal") #possible modes: "auto", "compact", "vertical", "minimal"
    
    #print("Plotting the partitions and dependencies of the model...")
    #plot_partitions(parts, deps)
    #print("Done!")
        
    task_graph = model_to_graph(model, grid, dep_graph, parts, deps, verbose=False)
    
    #plot_graph(task_graph)
    #print_dependencies(task_graph)
    
    ##### Optimization algorithms #####
    if ACO:
        print("\n ...Running Ant Colony Optimization...")
        # Redirect stdout to the Logger
        log_path = get_aco_log_path()
        sys.stdout = Logger(log_path)

        params = op.ACOParameters(
            n_ants = 10,
            rho = 0.05, #evaporation rate
            n_best = 3,
            n_iterations = 10,
            alpha = 1.,
            beta = 1.2,
        )
        n_procs = 2
        
        print(f"Creating the Ant Colony Optimization instance with {n_procs} processes running in parallel ants: {params.n_ants} for {params.n_iterations} iterations.")
        opt = op.AntColony( params, grid, task_graph, seed = None)

        #opt = op.ParallelAntColony(n_procs, params, grid, task_graph, seed = None)
        
        shortest = opt.run(once_every=1, show_traces= False)
        print("The best path found is: ")
        print(shortest)
        
        #imshow occupancy of the grid -> memory usage 
        
        #imshow occupancy of the gird -> resources usage over time
        #this guy probably using logger
        
        print("Visualizing the best path...\n")
        # Visualize the best path
        plot_timeline(path_to_json = get_ACO_DIR() + "/best_solution.json", timeline_path = get_ACO_DIR() + "/ACO_" + get_timestamp() + ".png", verbose = False)
            
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("\n ... End Ant Colony Optimization")
        print("Done!")

    if GA: 
        print("Running Genetic Algoriothm Optimization...")
        # Redirect stdout to the Logger
        log_path = get_ga_log_path()
        sys.stdout = Logger(log_path)
        
        params = op.GAParameters(
        sol_per_pop = 10, #512, #30,
        n_parents_mating= 5, #20, #Number of solutions to be selected as parents.
        keep_parents= -1 , #10, # -1 keep all parents, 0 means do not keep parents, 10 means 10 best parents etc
        parent_selection_type= "rws", # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection). 
        n_generations = 2, #800,
        mutation_probability = .4, #some exploration, so don’t kill mutation completely.
        crossover_probability = .9, #outlier genes to propagate = crossover must dominate.
        )
        
        # Note: actually the there is no function implemented to deal with the fact when out of all solutions no is feasible what to do, so it happens that after 1 iterations it gives rong solutions but after 10 of them is okay - check it later!
        #so there is a probablity that if you set n_parents_mating too big then is the problmer with final output, also sometimes the best solution given by  print(shortest[0], 1/shortest[1])it is not the best one
        
        n_procs = 10
        opt = op.GeneticAlgorithm(params, grid, task_graph, seed = None)
        #opt = op.ParallelGA(n_procs, params, grid, task_graph, seed = None)
                
        shortest = opt.run()
        #opt.ga_instance.plot_fitness()
        print("The best path found is: ")
        print(shortest[0], 1/shortest[1])
        
        print("Visualizing the best path...\n")
        # Visualize the best path
        plot_timeline(path_to_json = get_GA_DIR() + "/best_solution.json", timeline_path = get_GA_DIR() + "/GA_" + get_timestamp() + ".png", verbose = False)
        
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n ... End Genetic Algoriothm Optimization")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("Done!")



