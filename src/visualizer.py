'''
==================================================
File: visualizer.py
Project: simopty
File Created: Friday, 7th March 2025
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

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
import time
import nocsim # type: ignore

# Thanks to J. Jastrzebski
def create_logger(path_to_json = "/test.json", verbose = False):
    
    # Create a SimulatorStub object
    stub = ss.SimulatorStub()

    # Run the simulation in parallel
    
    # processors = list(range(6))
    # config_files = [os.path.join(RUN_FILES_DIR, f) for f in os.listdir(RUN_FILES_DIR) if f.endswith('.json')]
    # results, logger = stub.run_simulations_in_parallel(config_files=config_files, processors=processors, verbose=True)
    # results, logger = stub.run_simulation("config_files/dumps/dump.json", verbose = True)

    path_data = path_to_json
    os.makedirs(os.path.dirname(path_data), exist_ok=True)
    # append "logger" : 1, to the arch field of the json file while 
    # leaving the rest of the file unchanged
    
    with open(path_data, "r") as f:
        data = json.load(f)
        data["arch"]["logger"] = 1
    with open(path_data, "w") as f:
        json.dump(data, f, indent = 4)



    results, logger = stub.run_simulation(path_data, verbose = False)
    
    print("Latency: ", results)
    
    if verbose: 
        for event in logger.events:
            print(event)
            print(f"Event ID: {event.id}, Type: {event.type}, Cycle: {event.cycle}, Additional info: {event.additional_info}," 
                    f"Info: {event.info}")
            if event.type == nocsim.EventType.START_COMPUTATION:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"Node ID: {event.info.node} , Add_info Node ID {event.additional_info}")
            elif event.type == nocsim.EventType.END_COMPUTATION:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"Node ID: {event.info.node}, Add_info Node ID {event.additional_info}")
            elif event.type == nocsim.EventType.OUT_TRAFFIC:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"History: {event.info.history}")
            elif event.type == nocsim.EventType.IN_TRAFFIC:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"History: {event.info.history}")
            elif event.type == nocsim.EventType.START_RECONFIGURATION:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"Node ID: {event.additional_info}")
            elif event.type == nocsim.EventType.END_RECONFIGURATION:
                print(f"Type: {event.type}, Event info: {event.info}")
                print(f"Add_info Node ID: {event.additional_info}")
            else:
                print(f"I don't know how to handle this event: {event.type}")
                pass
    
    return logger, path_data

def plot_3d_animaiton(path_to_json = "/test.json", fps = 2, gif_path = "visual/test.gif"):
    
    logger, path_data = create_logger(path_to_json)
    
    # Initialize 3D plotter
    plotter_3d_animation = NoCPlotter()
    
    start_time = time.time()
    print("Plotting 3D animation...")
    plotter_3d_animation.plot(logger, fps, path_data, gif_path, verbose = False)  # Original 3D plot
    end_time = time.time()
    print(f"3D animation plotting took {end_time - start_time:.2f} seconds")

def plot_timeline(path_to_json = "/test.json", timeline_path = "visual/test.png", verbose = False):
    
    logger, path_data = create_logger(path_to_json)
    
    # Initialize timeline plotter
    plotter_timeline = NoCTimelinePlotter()

    print("Plotting timeline...")
    # Generate 2D timeline
    plotter_timeline.setup_timeline(logger, path_data)
    plotter_timeline.plot_timeline(timeline_path)
    
    if verbose:
        plotter_timeline._print_node_events()
    print("Timeline plotting done!")
    
def plot_timeline_factor_back(path_to_json = "/test.json", timeline_path = "visual/test.png", verbose = False):
    
    """
    Legacy code, to be deleted
    """
    
    logger, path_data = create_logger(path_to_json)
    
    # Initialize timeline plotter
    plotter_timeline = NoCTimelinePlotter()

    print("Plotting timeline...")
    # Generate 2D timeline
    plotter_timeline.setup_timeline(logger, path_data)
    plotter_timeline.plot_timeline_factor_back(timeline_path, factor_comp = 2.0 , factor_recon = 1.0)
    #factor_comp = 0.01 , factor_recon = 0.08
    if verbose:
        plotter_timeline._print_node_events()
    print("Timeline plotting factor back done!")
    

def plot_convergence(statistics_path, save_path=None):
    stats = np.load(os.path.join(statistics_path, "statistics.npy"), allow_pickle=True).item()
    mean_values = np.array(stats["mdn"])
    std_values = np.array(stats["std"])
    best_values = np.array(stats["best"])
    ab_best_values = np.array(stats["absolute_best"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    line_mean, = ax.plot(mean_values, label="iteration mean", color="lightseagreen")
    ax.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2, color="mediumturquoise", label="std deviation")
    line_best, = ax.plot(best_values, label="iteration best", color="lightcoral", linewidth=3)
    line_ab_best, = ax.plot(ab_best_values, label="absolute best", color="darkorange", linewidth=3, linestyle='--')

    ax.tick_params(direction='in')
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=14)
    
    plt.xlabel("Iterations", fontdict={"size": 14})
    plt.ylabel("Latency [cycles]", fontdict={"size": 14})
    plt.tight_layout()

    if save_path and isinstance(save_path, str):
        fig.savefig(save_path, dpi=300)
        print(f"Convergence graph saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        print("Convergence graph displayed.")
    
    
if __name__ == "__main__":
    # Example usage
    # plot_3d_animaiton(path_to_json = "/test.json", fps = 2, gif_path = "visual/test.gif")
    #plot_timeline(path_to_json = "data/data_mixed_part_seed_2137_2April25/ACO_seed_2137_x2_2025-04-08_13-47-07/best_solution.json",
                #timeline_path = "visual/seed_ACO_x2_refactored_from_test.png", verbose = False)
    #plot_timeline_factor_back(path_to_json = "data/ACO_onecyclerouter_x1_2025-04-28_14-10-49/best_solution.json",
    #                          timeline_path = "visual/ACO_router_oncecycle_x1.png", verbose = False)
    plot_timeline(path_to_json = "data/ACO_test_conv_1k_2025-06-20_21-13-40/best_solution.json", timeline_path = "visual/test_small_conv.png", verbose = False)
    #for i in range(0,29):
    #    plot_timeline(path_to_json = f"data/dump_GA_{i}.json", #timeline_path = "visual/test_ga_6.png", verbose = False)
    #plot_timeline(path_to_json = f"data/dump_GA_27.json", timeline_path = "visual/test_ga_6.png", verbose = False)
    #GA_seed_2137_x1_2025-04-02_00-00-20
    #GA_seed_2137_x25_2025-04-01_23-54-31