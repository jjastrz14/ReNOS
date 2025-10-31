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
from utils.simulator_utils import *
from graph import TaskGraph
from visualizer import plot_timeline, plot_convergence
from utils.model_fusion import fuse_conv_bn
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
from models import *


def count_operational_layers(model):
    """Count layers that need partitioning tuples"""
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ not in ['InputLayer']:
            count += 1
    return count


if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select optimization algorithm and name of the resulting directory.")
    parser.add_argument("-algo", choices=["ACO", "GA", "ACO_parallel", "GA_parallel"], required=True, help="Choose 'ACO' for Ant Colony Optimization or 'GA' for Genetic Algorithm.")
    parser.add_argument("-name", type=str, default="_run_", required=True, help="Choose name of the resulting directory (default: '_run_').")
    parser.add_argument("-partitioner-config", type=str, default=None, help="Path to the partitioner configuration file (optional).")
    parser.add_argument("-single-config", type=str, default=None, help="Single partitioning tuple for all layers in the model. Example: '(1,1,1)'")
    args = parser.parse_args()
    
    start = time.time()

    # Initialize global directories
    initialize_globals(args.algo, args.name)

    # Set flags based on the argument
    if args.algo == "ACO" or args.algo == "ACO_parallel":
        ACO = True
        GA = False
        # Redirect stdout to the Logger
        log_path = get_aco_log_path()
        sys.stdout = Logger(log_path)
        
    elif args.algo == "GA" or args.algo == "GA_parallel":
        ACO = False
        GA = True
        # Redirect stdout to the Logger
        log_path = get_ga_log_path()
        sys.stdout = Logger(log_path)
        
    print(f"Selected algorithm: {args.algo}")
        
    # Get the shared timestamp
    # Debug - Print the global variables to check they're set correctly
    print(f"After initialization:")
    debug_globals()
    
    #measute time of the optmiization
    #model = LeNet4((28, 28, 1), verbose=True)
    #model = single_conv((10, 10, 4), num_classes=1, verbose=True)
    # model = Resnet9s((32, 32, 3), verbose=True)
    # model = test_conv((28, 28, 1), num_classes = 2, verbose=True)
    # model = test_model((28, 28, 1), verbose= True)
    # model = small_test_model((28, 28, 1))
    # model = load_model("ResNet50")
    # model = load_model("MobileNet")
    # model = load_model("MobileNetV2")
    #model = ResNet32_early_blocks((32, 32, 3), verbose=True)
    #model = VGG_early_layers((32, 32, 3), verbose=True)
    #model = VGG_late_layers((14, 14, 512), verbose=True)
    
    #model = AlexNet((32, 32, 3), num_classes=10, verbose=True) #[(1, 2, 3)]
    #model = MobileNetv1((32, 32, 3), num_classes=10, verbose=True) #[(0, 2, 3)]
    #model = ResNet32_early_blocks((32, 32, 3), verbose=True) #[(0, 2, 4)]
    #model = ResNet32_mid_blocks((32, 32, 16), num_classes=10, verbose=True) #[(0, 2, 5)]
    #model = ResNet32_late_blocks(input_shape=(16, 16, 32), num_classes=10, verbose=False) #[(1, 3, 3)]
    #model = VGG_16_early_layers((32, 32, 3), num_classes=10, verbose=True) #[(1, 2, 1)]
    #model = VGG_16_late_layers((4, 4, 256), num_classes=10, verbose=True) #[(0, 1, 3)]
    
    model = ResNet_block_smaller(input_shape=(56, 56, 32), num_classes=10, verbose=True)
    
    model = fuse_conv_bn(model, verbose=True)
    
    num_layers = count_operational_layers(model)
    print(f"\nModel has {num_layers} operational layers requiring tuples\n")
    
    if args.partitioner_config:
        print(f"Loading configuration from {args.partitioner_config} JSON file...\n")

        try:
            # Load JSON file
            with open(args.partitioner_config, 'r') as f:
                data = json.load(f)

            # Check if it has 'configurations' array
            if 'configurations' not in data:
                raise ValueError(
                    "Invalid JSON format. Expected format:\n"
                    '{\n'
                    '  "configurations": [\n'
                    '    {\n'
                    '      "metric": "size",\n'
                    '      "target_value": 5000,\n'
                    '      "configuration": {"1": [5,1,4], "2": [4,4,2], ...}\n'
                    '    }\n'
                    '  ]\n'
                    '}'
                )

            if not isinstance(data['configurations'], list):
                raise ValueError("'configurations' must be an array")

            if len(data['configurations']) == 0:
                raise ValueError("'configurations' array is empty")

            # Extract the single configuration
            config = data['configurations'][0]

            if 'configuration' not in config:
                raise ValueError("Configuration object missing 'configuration' key with layer configs")

            layer_configs = config['configuration']

            print(f"Detected configurations array format")
            print(f"  Configuration: {config.get('metric', 'unknown')} = {config.get('target_value', 'N/A')}")

            # Build partitioner tuples
            partitioner_tuples = build_partitioner_tuples_for_model(model, layer_configs, verbose=False)

            # Extract metadata
            metadata = {
                'metric': config.get('metric', ''),
                'target_value': config.get('target_value', 0),
                'tolerance': config.get('tolerance', 0),
                'max_partitions_per_layer': config.get('max_partitions_per_layer', 0),
                'avg_flops': config.get('avg_flops', 0),
                'avg_size': config.get('avg_size', 0),
                'avg_flops_no_relu': config.get('avg_flops_no_relu', 0),
                'avg_size_no_relu': config.get('avg_size_no_relu', 0)
            }

            print(f"\n{'='*80}")
            print(f"Configuration loaded:")
            print(f"  Metric: {metadata['metric']}")
            print(f"  Target Value: {metadata['target_value']:.2e}")
            print(f"  Avg FLOPs: {metadata['avg_flops']:.2e}")
            print(f"  Avg Size: {metadata['avg_size']:,.0f}")
            print(f"  Partitioner tuples: {partitioner_tuples}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"✗ Error loading {args.partitioner_config}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    elif args.single_config:
        single_tuple = tuple(map(int, args.single_config.strip("()").split(",")))
        partitioning_per_layer = [single_tuple] * num_layers
        partitioner_tuples = [(0, 1, 1)] + partitioning_per_layer
        num_partitions = 2**partitioner_tuples[1][0]*partitioner_tuples[1][1]*partitioner_tuples[1][2]
        print(f"\nUsing single partitioning tuple {single_tuple} for all layers.")
        
    else:
        print("WARN: No partitioner configuration file or single tuple provided. Using default partitioning.")
        partitioning_per_layer = [(1, 3, 3)]
        partitioner_tuples = [(0, 1, 1)] + partitioning_per_layer * (len(model.layers))
        num_partitions = 2**partitioner_tuples[1][0]*partitioner_tuples[1][1]*partitioner_tuples[1][2]
    
    # grid is: number of processor x number of processors (size_of_grid x size_of_grid)
    size_of_grid = 12
    source = 0
    drain = 143

    grid = dm.Grid()
    grid.init(size_of_grid, 2, dm.Topology.TORUS, source = source, drain = drain)
    
    #### Model analysis and partitioning ####

    print("")
    print("Analysis of the model...")
    analyze_ops(model, incl_info = True)
        
    dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
    #spatial, output, input
    #parts, deps = build_partitions(model, grid, chosen_splitting_strategy = "input", grouping = False, verbose = True)

        
    dep_graph = TaskGraph(source = grid.source, drain = grid.drain)
    #spatial, output, input
    parts, deps = build_partitions_splitting_input_for_many_tuples(model, grid, partitioning_tuple = partitioner_tuples, grouping = False, verbose = True)
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

        #params = op.ACOParameters(
        #    n_ants = 512, #512,
        #    rho = 0.05, #evaporation rate
        #    n_best = 100,
        #    n_iterations = 1000,
        #    alpha = 1.,
        #    beta = 1.2,
        #    is_analytical = True, #use analytical model instead of cycle-accurate simulator
        #    start_row_wise= False, #start from a row-wise mapping
        #    early_stop = 150
        #)
        #n_procs = 128
        
        params = op.ACOParameters(
            n_ants = 50, #512,
            rho = 0.05, #evaporation rate
            n_best = 25,
            n_iterations = 1000,
            alpha = 1.,
            beta = 1.2,
            is_analytical = True, #use analytical model instead of cycle-accurate simulator
            start_row_wise= False, #start from a row-wise mapping
            early_stop = 10
        )
        n_procs = 5
        
        if args.algo == "ACO_parallel":
            print(f"Creating the Parallel Ant Colony Optimization instance with {n_procs} processes running in parallel ants: {params.n_ants} for {params.n_iterations} iterations.")
            opt = op.ParallelAntColony(n_procs, params, grid, task_graph, seed = None)
        else:
            print(f"Creating the Ant Colony Optimization instance with ants: {params.n_ants} for {params.n_iterations} iterations.")
            opt = op.AntColony( params, grid, task_graph, seed = None)


        shortest = opt.run(once_every=1, show_traces= False)
        print("The best path found is: ")
        print(shortest)
        
        plot_convergence(get_ACO_DIR(), save_path=get_ACO_DIR() + "/convg_ACO_" + get_timestamp() + ".png")
        
        # Compare simulators and visualize results
        #results = compare_simulators_and_visualize(
        #    best_solution_path=get_ACO_DIR() + "/best_solution.json",
        #    output_dir=get_ACO_DIR(),
        #    algorithm_name="ACO",
        #    timestamp=get_timestamp(),
        #    verbose=False
        #)
            
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("\n ... End Ant Colony Optimization")
        print("Done!")

    if GA: 
        print("Running Genetic Algorithm Optimization...")
        
        #params = op.GAParameters(
        #sol_per_pop = 512, #30,
        #n_parents_mating= 100, #Number of solutions to be selected as parents.
        #keep_parents= -1 , #10, # -1 keep all parents, 0 means do not keep parents, 10 means 10 best parents etc
        #parent_selection_type= "sss", # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection). k = 3 for tournament, can be changed
        #n_generations = 1000, #800,
        #mutation_probability = .4, #some exploration, so don’t kill mutation completely.
        #crossover_probability = .9, #outlier genes to propagate = crossover must dominate.
        #is_analytical = True, #use analytical model instead of cycle-accurate simulator
        #start_row_wise= False, #start from a row-wise mapping
        #early_stop = 150
        #)
        
        #n_procs = 128
        
        params = op.GAParameters(
        sol_per_pop = 50, #30,
        n_parents_mating= 25, #Number of solutions to be selected as parents.
        keep_parents= -1 , #10, # -1 keep all parents, 0 means do not keep parents, 10 means 10 best parents etc
        parent_selection_type= "sss", # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection). k = 3 for tournament, can be changed
        n_generations = 1000, #800,
        mutation_probability = .4, #some exploration, so don’t kill mutation completely.
        crossover_probability = .9, #outlier genes to propagate = crossover must dominate.
        is_analytical = True, #use analytical model instead of cycle-accurate simulator
        start_row_wise= False, #start from a row-wise mapping
        early_stop = 150
        )
        
        n_procs = 5
        
        if args.algo == "GA_parallel":
            print(f"Creating the Parallel Genetic Algorithm instance with {n_procs} processes, population size: {params.sol_per_pop}, generations: {params.n_generations}.")
            opt = op.ParallelGA(n_procs, params, grid, task_graph, seed = None)
        else:
            print(f"Creating the Genetic Algorithm instance with, population size: {params.sol_per_pop}, generations: {params.n_generations}.")
            opt = op.GeneticAlgorithm(params, grid, task_graph, seed = None)
                
        shortest = opt.run()
        #opt.ga_instance.plot_fitness()
        print("The best path found is: ")
        print(shortest[0], shortest[1])
        
        #opt.summary()
        
        plot_convergence(str(get_GA_DIR()), save_path=get_GA_DIR() + "/convg_GA_" + get_timestamp() + ".png")
        
        # Compare simulators and visualize results
        #results = compare_simulators_and_visualize(
        #    best_solution_path=get_GA_DIR() + "/best_solution.json",
        #    output_dir=get_GA_DIR(),
        #    algorithm_name="GA",
        #    timestamp=get_timestamp(),
        #    verbose=False  #always creates timeline visualization
        #)
        
        #opt.plot_summary()
        
        end = time.time()
        elapsed_time = end - start
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n ... End Genetic Algoriothm Optimization")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
        print("Done!")



