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



if __name__ == '__main__':
        
    date = "22Sept_data"
    path = "./data/22Sept_data"
    model_name = "model_resnet_early_blocks"
    strategy= 'GA'
    num_partitions=16
    x_of_grid = 12
    topology_name = 'Torus'
    
    # Measure Booksim2 simulation time
    print("Booksim2 simulation...")
    stub = ss.SimulatorStub()
    start_time = time.time()
    result, logger = stub.run_simulation(f"{path}/best_solution_{strategy}.json", dwrap = False, verbose = False)
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