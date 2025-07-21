'''
==================================================
File: optimizer.py
Project: simopty
File Created: Wednesday, 18th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The optimizer.py module contains the classes used to perform the optimization of the mapping of a split-compiled NN onto a k-dimensional NoC grid.
"""

import os, sys
import logging
import enum
import numpy as np
import random
from numpy.random import seed
import simulator_stub as ss
import mapper as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Union, ClassVar, Optional
from dirs import get_CONFIG_DUMP_DIR, get_ARCH_FILE, get_ACO_DIR
from utils.plotting_utils import plot_mapping_gif
import ctypes as c
from contextlib import closing
import multiprocessing as mp
import concurrent.futures
from utils.aco_utils import Ant, vardict, walk_batch, update_pheromones_batch, manhattan_distance, random_heuristic_update
from utils.partitioner_utils import PE


class OptimizerType(enum.Enum):
    """
    Enum class representing the type of optimizer.
    """
    ACO = "ACO"
    GA = "GA"


@dataclass
class ACOParameters:
    """
    Dataclass representing the parameters of the optimization algorithm.

    The parameters of the optimization algorithm:
            - alpha : float
                The alpha parameter of the ACO algorithm.
            - beta : float
                The beta parameter of the ACO algorithm.
            - rho : float
                The evaporation rate of the pheromone.
            - n_ants : int
                The number of ants fro each colony
            - n_iterations : int
                The number of iterations of the algorithm.
            - n_best : int
                The number of best solutions to keep track of.
            - starting_rho : float
    """
    alpha : float = 1.0
    beta : float = 1.0
    rho : Union[float, None] = None
    n_ants : int = 10
    n_iterations : int = 100
    n_best : int = 10
    rho_start : Union[float, None] = None
    rho_end : Union[float, None] = None


class BaseOpt :

    optimizerType : ClassVar[Optional[Union[str, OptimizerType]]] = None

    def __str__(self):
        return self.optimizerType.value
    
    def run(self):
        pass
    

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ACO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
"""
class AntColony(BaseOpt):

    optimizerType : ClassVar[Union[str, OptimizerType]] = OptimizerType.ACO

    def __init__(self, optimization_parameters, domain, task_graph, seed = None):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__()
        self.par = optimization_parameters

        # --- Domain and Task Graph ---
        self.domain = domain
        self.task_graph = task_graph

        # --- Pheromone and Heuristic Information ---
        # The DAG on which to optimize is made out of implementation points, particular combinations
        # of tasks and PEs, representig to which PE a task is mapped to.
        #  the number of nodes in this new graph = #tasks * #PEs

        self.tau_start = np.ones((domain.size))/ domain.size
        #self.tau_end = np.ones((domain.size))/ domain.size
        #Question: if there is tau_start for the start node, should we add also a tau_end for the end node or it is enough to have this in tau matirx?
        
        self.tau = np.ones((task_graph.n_nodes, domain.size, domain.size))/ (task_graph.n_nodes * domain.size * domain.size) # Pheromone matrix
        self.eta = np.ones((task_graph.n_nodes, domain.size, domain.size)) # Heuristic matrix

        #create a list of task ID ensuring the start node is the first element and end node is the last
        tasks = [task["id"] for task in self.task_graph.get_nodes()] 
        
        self.tasks = tasks
                
        #seed
        self.seed = seed
        
        #statistics measure
        self.statistics = {}
        self.statistics["mdn"] = []
        self.statistics["std"] = []
        self.statistics["best"] = []
        self.statistics["absolute_best"] = [np.inf]
        
        self.ACO_DIR = get_ACO_DIR()
        self.CONFIG_DUMP_DIR = get_CONFIG_DUMP_DIR()
        self.ARCH_FILE = get_ARCH_FILE()



    def run_and_show_traces(self, single_iteration_func, **kwargs):
        """
        This function is used to show the most likely paths taken by the ants throughout the 
        execution of the algorithm.
        It produces a gif file, evaluating the pheromone traces at a certain iteration and 
        finally plotting those, then stiching them up to create the animation
        
        Legacy animation function, not used anymore - need adjustments to work with the new ACO implementation
        """

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("magma")
        margin = 0.5
        ax.axis("off")
        ax.set_xlim( - 1 - margin, len(self.tasks) + margin)
        ax.set_ylim( - margin , self.domain.size + margin)

        a = 0.7 # ellipse width
        b = 0.5 # ellipse height    

        #implementation_points = []
        for i,task in enumerate(self.tasks[1:]):
            for j in range(self.domain.size):
                #implementation_points.append((i, j))
                ax.add_patch(patches.Ellipse((i, j), a, b, alpha = 1, facecolor = 'navajowhite', edgecolor = 'black', linewidth = 2, zorder = 2))
                ax.text(i, j, "(%s, %d)" % (task, j), ha = 'center', va = 'center', color = 'black')
        
        # Add starting and ending points
        ax.add_patch(patches.Ellipse((-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'palegreen', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(-1, int(self.domain.size/2), "start", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        ax.add_patch(patches.Ellipse((len(self.tasks)-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'lightcoral', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(len(self.tasks)-1, int(self.domain.size/2), "end", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        
        # Draw the connections (fully connected layers style)
        for i in range(len(self.tasks)-2):
            for j in range(self.domain.size):
                for k in range(self.domain.size):
                    ax.plot([i, i+1], [j, k], color = 'lightgrey', zorder = -2)

        for j in range(self.domain.size):
            ax.plot([-1, 0], [int(self.domain.size/2),j], color = 'lightgrey', zorder = -2)
            ax.plot([len(self.tasks)-2, len(self.tasks)-1], [j, int(self.domain.size/2)], color = 'lightgrey', zorder = -2)

        all_time_shortest_path = []
        edges = []

        def update(frame, edges = edges, all_time_shortest_path = all_time_shortest_path):
            # Each time the update function is called, an iteration of the algorithm is performed
            # and, if the frame is a multiple of the once_every parameter, the plot is actually updated

            all_time_shortest_path.append(single_iteration_func(frame, kwargs["once_every"], kwargs["rho_step"]))
            if len(all_time_shortest_path) > 1:
                assert len(all_time_shortest_path) == 2
                if all_time_shortest_path[1][1] < all_time_shortest_path[0][1]:
                    all_time_shortest_path[0] = all_time_shortest_path[1]
                all_time_shortest_path.pop(1)


            if frame % kwargs["once_every"] == 0:
                # extract the pheromone values and plot them on the corresponding edges
                

                for edge in edges:
                    edge.remove()
                edges.clear()

                for d_level in range(self.tau.shape[0]):
                    vmax = np.max(self.tau[d_level])
                    for i in range(self.tau.shape[1]):
                        for j in range(self.tau.shape[2]):
                            # find the maximum value out of the pheromones of che considered level
                            if self.tau[d_level, i, j] > 0:
                                edges.append(ax.plot([d_level, d_level+1], [i, j], color = cmap(self.tau[d_level, i, j]/vmax), zorder = self.tau[d_level, i, j]/vmax)[0])
                
                vmax_start = np.max(self.tau_start)
                for i in range(self.tau_start.shape[0]):
                    if self.tau_start[i] > 0:
                        edges.append(ax.plot([-1, 0], [int(self.domain.size/2), i], color = cmap(self.tau_start[i]/vmax_start), zorder = self.tau[d_level, i, j]/vmax)[0])

                return edges

            return []
                

        #ani = animation.FuncAnimation(fig, update, frames = kwargs["n_iterations"], repeat = False)
        #path = os.path.join(self.ACO_DIR, "pheromone_traces.png")
        #plt.savefig(path, dpi = 100)
        #plt.close()

        return all_time_shortest_path

    def run(self, once_every = 10, show_traces = False):
        """
        Run the algorithm
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)


        def single_iteration(i, once_every, rho_step = 0):
            all_paths = self.generate_colony_paths()
            self.update_pheromones(all_paths)
            # print(self.tau)
            # plot a heatmap of the pheromonesa at dlevel 0 and save it
            # fig, ax = plt.subplots()
            # cmap = plt.get_cmap("magma")
            # slice = self.tau[0]
            # vmax = np.max(slice)
            # ax.imshow(slice, cmap = cmap, vmax = vmax)
            # plt.show()
            # plt.close()

            self.update_heuristics()
            shortest_path = min(all_paths, key=lambda x: x[2])
            moving_average = np.mean([path[2] for path in all_paths])
            moving_std = np.std([path[2] for path in all_paths])
            if once_every is not None and i%once_every == 0:
                number, path_list, value = shortest_path
                print(f"Iteration # {i} chosen path is: {number}, {path_list[0:5]}, ..., {path_list[-5:]}, latency: {value}")
                print("Moving average for the path lenght is:", moving_average)
            
            self.evaporate_pheromones(rho_step)
            self.statistics["mdn"].append(moving_average)
            self.statistics["std"].append(moving_std)
            self.statistics["best"].append(shortest_path[2])
            if shortest_path[2] < self.statistics["absolute_best"][-1]:
                self.statistics["absolute_best"].append(shortest_path[2])
            else:
                self.statistics["absolute_best"].append(self.statistics["absolute_best"][-1])
            return shortest_path

        shortest_path = None
        #correction to count also ants
        all_time_shortest_path = (np.inf, "placeholder", np.inf)

        if self.par.rho_start is not None and self.par.rho_end is not None:
            self.rho = self.par.rho_start
            rho_step = (self.par.rho_end - self.par.rho_start) / self.par.n_iterations
        else:
            self.rho = self.par.rho
            rho_step = 0

        if show_traces:
            all_time_shortest_path = self.run_and_show_traces(single_iteration, once_every = once_every, n_iterations = self.par.n_iterations, rho_step = rho_step)
        else:
            for i in range(self.par.n_iterations):
                shortest_path = single_iteration(i, once_every, rho_step)
                if shortest_path[2] < all_time_shortest_path[2]:
                    all_time_shortest_path = shortest_path 
                    
                    ant_int = str(shortest_path[0])
                    #check if globals are initialized
                    if not self.CONFIG_DUMP_DIR or not self.ACO_DIR:
                        raise RuntimeError("Directories not initialized. Call initialize_globals() first.")

                    dump_file = os.path.join(self.CONFIG_DUMP_DIR, f"dump{ant_int}.json")
                    #dump_file = CONFIG_DUMP_DIR + "/dump" + ant_int + ".json"
                    print(f"Saving the best solution found by ant {ant_int} in" + self.ACO_DIR + "/best_solution.json")
                    
                    #save the corresponding dump file into data
                    os.system(f"cp {dump_file} {self.ACO_DIR}")
                    #save the dump of the best solution in data
                    os.system(f"mv {self.ACO_DIR}/dump{ant_int}.json {self.ACO_DIR}/best_solution.json")
                    #os.system("mv " + ACO_DIR + "/dump" + ant_int + ".json " + ACO_DIR + "/best_solution.json")
                np.save(self.ACO_DIR + "/statistics.npy", self.statistics)
                print(f"Iteration {i} done. Saving statistics in: " + self.ACO_DIR)
                print("-" * 50 + "\n")
                    
            # Finalize the simulation: save the data
            np.save(self.ACO_DIR + "/statistics.npy", self.statistics)
            print("Saving Final Results in: " + self.ACO_DIR)
            print(" ")
        
        return all_time_shortest_path


    def pick_move(self,task_id, d_level, current, resources, added_space, prev_path, random_heuristic = False):
        """
        Pick the next move of the ant, given the pheromone and heuristic matrices.
        if random_heuristic chosen to False then the heuristic is the manhattan distance!
        """
        
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # compute a mask to filter out the resources that are already used
        mask = np.array([0 if pe.mem_used + added_space > pe.mem_size else 1 for pe in resources])

        if d_level == 0:
            #init tau start
            outbound_pheromones = self.tau_start
            outbound_heuristics = np.ones(self.domain.size)
        else:
            #next step taus and pheromonoes
            outbound_pheromones = self.tau[d_level-1,current,:]
            if random_heuristic:
                outbound_heuristics = self.eta[d_level-1, current, :]
            else:
                #tau updated as manhattan distance
                # find the id of the task on which task_id depends (may be multiple)
                dependencies = self.task_graph.get_dependencies(task_id)
                # print("The dependencies of the task are:", dependencies)
                if len(dependencies) == 0:
                    outbound_heuristics = self.eta[d_level-1, current, :]
                else:
                    # find the PE on which the dependencies are mapped
                    dependencies_pe = [pe[2] for pe in prev_path if pe[0] in dependencies]
                    # print("The PEs on which the dependencies are mapped are:", dependencies_pe)
                    # generate the heuristics to favour the PEs near the  ones where the dependencies are mapped
                    outbound_heuristics = np.zeros(self.domain.size)
                    for pe in dependencies_pe:
                        for i in range(self.domain.size):
                            outbound_heuristics[i] += 1 / manhattan_distance(pe, i, self.domain) if manhattan_distance(pe, i, self.domain) != 0 else 1
                    outbound_heuristics = outbound_heuristics / np.sum(outbound_heuristics)
                

        row = (outbound_pheromones ** self.par.alpha) * (outbound_heuristics ** self.par.beta) * mask
        norm_row = (row / row.sum()).flatten()

        # if there is a NaN in the row, raise an error
        if np.isnan(norm_row).any():
            # print the row value that caused the error
            print("The row is:", row)
            raise ValueError("The row is NaN")
        
        return np.random.choice(range(self.domain.size), 1, p = norm_row)[0]


    def generate_ant_path(self, verbose = False):
        """
        Generate a path for the ant based on pheromone and heuristic matrices.
        
        The path consists of tuples (task_id, current_node, next_node) representing:
        - The task being processed
        - The node where the task starts
        - The node where the task will be executed
        
        The ant starts from a source node and ends at a drain node defined in the task graph.
        The first entry has current_node=-1 and next_node=source_node.
        
        Returns:
            list: A path represented as a list of (task_id, current_node, next_node) tuples
        """
        
        #initilaize the path and resource tracking
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        # A list of the available resources for each PE
        resources = [PE() for _ in range(self.domain.size)]
        
        # the last node is decalred as drain point and the starting point is source point
        source_node = self.task_graph.SOURCE_POINT
        drain_node = self.task_graph.DRAIN_POINT
        
        #start with previous node as -1 (no previous node yet)
        prev_node = -1
    
        for d_level, task_id in enumerate(self.tasks):
            current_node = prev_node
            
            #determine resources requirmnets for this task
            if task_id not in ("start", "end"):
                task_size = self.task_graph.get_node(task_id)["size"]
            else:
                task_size = 0
            
            #Handle special case for start and end tasks
            if task_id == "start":
                next_node = source_node
            #case to map last on the drain node
            
            elif task_id == "end": #case to connect last to "end"
                next_node = drain_node 
            else:
                # Pick the next node based on pheromone, heuristic, and resource availability
                next_node = self.pick_move(task_id, d_level, current_node, resources, task_size, path)
            
            # udpate the resources
            if task_id != "start" and task_id != "end":
                resources[next_node].mem_used += task_size
            
            #normal case
            path.append((task_id, current_node, next_node))
            prev_node = next_node

        if verbose:
            print("The path found by the ant is:", path)
        return path


    def path_length(self, ant_id, path, verbose = False):
        """
        Compute the "length" of the path using the NoC simulator.
        """
        if not self.CONFIG_DUMP_DIR:
            raise RuntimeError("Config Dump dir not initialized. Call initialize_globals() first.")

        # constuct the mapping form the path
        mapping = {task_id : int(pe) for task_id, pe, _ in path if task_id != "start" and task_id != "end"}

        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(self.CONFIG_DUMP_DIR + "/dump{}.json".format(ant_id), file_to_append=self.ARCH_FILE)
        
        if verbose:
            plot_mapping_gif(mapper, "../visual/solution_mapping.gif")

        stub = ss.SimulatorStub()
        result, logger = stub.run_simulation(self.CONFIG_DUMP_DIR + "/dump{}.json".format(ant_id), dwrap=True)
        #result is number of cycles of chosen path and logger are the events one by one hapenning in restart
        return result, logger


    def generate_colony_paths(self):
        colony_paths = []
        for _ in range(self.par.n_ants):
            ant_path = self.generate_ant_path()
            #pass and id and ant_path to restart to measure the path length
            path_length = self.path_length( _ , ant_path)
            #append ant_id, ant_path and length of this path
            colony_paths.append((_ , ant_path, path_length[0]))
        return colony_paths
    
    def evaporate_pheromones(self, step):
        if self.par.rho is not None:
            self.par.rho += step
        else:
            raise ValueError("The evaporation rate is not set")
        self.tau_start = (1 - self.rho) * self.tau_start
        self.tau = (1 - self.rho) * self.tau

    def update_pheromones(self, colony_paths):
        if self.par.n_best is None:
            self.par.n_best = len(colony_paths)
        sorted_paths = sorted(colony_paths, key = lambda x : x[1])
        best_paths = sorted_paths[:self.par.n_best]
        
        for ant_id, path, path_length in best_paths:
            for d_level, path_node in enumerate(path):
                if path_node[1] == -1: # starting decision level
                    self.tau_start[path_node[2]] += 1 / path_length
                elif d_level-1 < self.tau.shape[0]:
                    self.tau[d_level-1, path_node[1], path_node[2]] += 1 / path_length
                else:
                    raise ValueError("The path node is not valid")

    def update_heuristics(self):
        """
        Update the heuristic matrix.
        """
        # RANDOM HEURISTIC UPDATE
        self.eta = random_heuristic_update(self.task_graph, self.domain)
    

class ParallelAntColony(AntColony):

    optimizerType : ClassVar[Union[str, OptimizerType]] = OptimizerType.ACO


    def __init__(self, number_of_processes, optimization_parameters, domain, task_graph, seed = None):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__(optimization_parameters, domain, task_graph, seed)

        # The number of executors that will be used to run the algorithm
        self.n_processes = number_of_processes

        # self.logger = mp.log_to_stderr()
        # self.logger.setLevel(logging.INFO)

        self.ants = [Ant(i, self.task_graph, self.domain, self.tasks, self.par.alpha, self.par.beta) for i in range(self.par.n_ants)]
        
        #seed for repeating simulations
        self.seed = seed

        assert self.n_processes <= self.par.n_ants, "The number of processes cannot exceed the number of ants"
        # --- Pheromone and Heuristic Information ---
        # The pheromone and heuristic matrices are shared arrays among the ants

        self.tau_start = mp.Array(c.c_double, self.domain.size)
        tau_start_np = np.frombuffer(self.tau_start.get_obj())
        #initialize the tau_start vector
        tau_start_np[:] = 1 / self.domain.size

        self.tau = mp.Array(c.c_double, (self.task_graph.n_nodes-1) * self.domain.size * self.domain.size)
        tau_np = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        #initialize the tau tensor
        tau_np[:] = 1 / (self.domain.size * self.domain.size * self.task_graph.n_nodes)

        self.eta = mp.Array(c.c_double, (self.task_graph.n_nodes-1) * self.domain.size * self.domain.size)
        eta_np = np.frombuffer(self.eta.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        #initialize the eta tensor
        eta_np[:] = 1

        self.statistics = {}
        self.statistics["mdn"] = []
        self.statistics["std"] = []
        self.statistics["best"] = []
        self.statistics["absolute_best"] = [np.inf]

        #Calculate and store intervals for parallel processing
        
        self.intervals = [ (i, i + self.par.n_ants//self.n_processes + min(i, self.par.n_ants % self.n_processes)) for i in range(0, self.par.n_ants, self.par.n_ants//self.n_processes)]
        if self.par.n_best >= self.n_processes:
            self.best_intervals = [ (i, i + self.par.n_best//self.n_processes + min(i, self.par.n_best % self.n_processes)) for i in range(0, self.par.n_best, self.par.n_best//self.n_processes)]

    
    def run(self, once_every = 10, show_traces = False):
        """
        Run the algorithm
        """
        
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        def single_iteration(i, once_every, rho_step = 0):
            all_paths = self.generate_colony_paths()
            self.update_pheromones(all_paths)

            # # plot a heatmap of the pheromonesa at dlevel 0
            # fig, ax = plt.subplots()
            # cmap = plt.get_cmap("magma")
            # slice = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)[0]
            # vmax = np.max(slice)
            # print("Vmax:", vmax)
            # ax.imshow(slice, cmap = cmap, vmax = vmax)
            # plt.show()
            # plt.close()

            self.update_heuristics()
            shortest_path = min(all_paths, key=lambda x: x[2])
            moving_average = np.mean([path[2] for path in all_paths])
            moving_std = np.std([path[2] for path in all_paths])
            if once_every is not None and i%once_every == 0:
                number, path_list, value = shortest_path
                print(f"Iteration # {i} chosen path is: {number}, {path_list[0:5]}, ..., {path_list[-5:]}, latency: {value}")
                print("Moving average for the path lenght is:", moving_average)
            
            self.evaporate_pheromones(rho_step)
            self.statistics["mdn"].append(moving_average)
            self.statistics["std"].append(moving_std)
            self.statistics["best"].append(shortest_path[2])
            if shortest_path[2] < self.statistics["absolute_best"][-1]:
                self.statistics["absolute_best"].append(shortest_path[2])
            else:
                self.statistics["absolute_best"].append(self.statistics["absolute_best"][-1])
            return shortest_path


        shortest_path = None
        all_time_shortest_path = (np.inf, "placeholder", np.inf)

        if self.par.rho_start is not None and self.par.rho_end is not None:
            self.rho = self.par.rho_start
            rho_step = (self.par.rho_end - self.par.rho_start) / self.par.n_iterations
        else:
            self.rho = self.par.rho
            rho_step = 0

        if show_traces:
            all_time_shortest_path = self.run_and_show_traces(single_iteration, once_every = once_every, n_iterations = self.par.n_iterations, rho_step = rho_step)
        else:
            for i in range(self.par.n_iterations):
                shortest_path = single_iteration(i, once_every, rho_step)
                if shortest_path[2] < all_time_shortest_path[2]:
                    all_time_shortest_path = shortest_path 
                    # save the dump of the best solution in data
                    # 1. get the id of the ant that found the best solution
                    ant_id = shortest_path[0]
                    # save the corresponding dump file into data
                    if not self.CONFIG_DUMP_DIR or not self.ACO_DIR:
                        raise RuntimeError("Directories not initialized. Call initialize_globals() first.")
                
                    dump_file = os.path.join(self.CONFIG_DUMP_DIR, f"dump{ant_id}.json")
                    
                    print("Saving the best solution found by ant", ant_id, "in " + self.ACO_DIR + "/best_solution.json")
                    #save the corresponding dump file into data
                    os.system(f"cp {dump_file} {self.ACO_DIR}")
                    #save the dump of the best solution in data
                    os.system(f"mv {self.ACO_DIR}/dump{ant_id}.json {self.ACO_DIR}/best_solution.json")
                np.save(self.ACO_DIR + "/statistics.npy", self.statistics)
                print(f"Iteration {i} done. Saving statistics in: " + self.ACO_DIR)
                print("-" * 50 + "\n")
                    
            # Finalize the simulation: save the data
            np.save(self.ACO_DIR + "/statistics.npy", self.statistics)
            print("Saving Final Results in: " + self.ACO_DIR)
            print(" ")
            

        return all_time_shortest_path

    @staticmethod
    def init(tau_start_, tau_, eta_, tau_shape, eta_shape, **kwargs):
        global vardict
        vardict["tau_start"] = tau_start_
        vardict["tau"] = tau_
        vardict["eta"] = eta_
        vardict["tau.size"] = tau_shape
        vardict["eta.size"] = eta_shape

    def generate_colony_paths(self):
        # generate the colony of ants (parallel workers)
        
        with closing(mp.Pool
                    (processes = self.n_processes, 
                    initializer = ParallelAntColony.init, 
                    initargs = (self.tau_start, self.tau, self.eta, 
                                (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), 
                                (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)),
                                )) as pool:
            # generate the paths in parallel: each process is assigned to a subset of the ants
            # evenly distributed
            # seed below allows for setting same seed for each ants (set it by passing seed to the constructor)
            colony_paths = pool.map_async(walk_batch,[(self.ants[start:end], self.seed)for start, end in self.intervals])
            colony_paths = colony_paths.get()
        pool.join()
        # unpack the batches of paths
        colony_paths = [path for batch in colony_paths for path in batch]
        return colony_paths
    
    
    def evaporate_pheromones(self, step):
        if self.par.rho is not None:
            self.par.rho += step
        else:
            raise ValueError("The evaporation rate is not set")
        tau_start = np.frombuffer(self.tau_start.get_obj())
        tau_start[:] = (1 - self.rho) * tau_start
        tau = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        tau[:] = (1 - self.rho) * tau


    def update_pheromones(self, colony_paths):
        if self.par.n_best is None:
            self.par.n_best = len(colony_paths)
        sorted_paths = sorted(colony_paths, key = lambda x : x[2])
        best_paths = sorted_paths[:self.par.n_best]
        if self.par.n_best < self.n_processes:
            # update the pheromones in parallel
            with closing(mp.Pool(processes = self.par.n_best, initializer = ParallelAntColony.init, initargs = (self.tau_start, self.tau, self.eta, (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)))) as pool:
                pool.map(update_pheromones_batch, [[best_paths[i]] for i in range(self.par.n_best)])
            pool.join()
        else:
            # update the pheromones in parallel
            with closing(mp.Pool(processes = self.n_processes, initializer = ParallelAntColony.init, initargs = (self.tau_start, self.tau, self.eta, (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)))) as pool:
                pool.map(update_pheromones_batch, [best_paths[start:end] for start, end in self.best_intervals])
            pool.join()

    def update_heuristics(self): 
        # introduce stochasticity in the heuristic matrix
        eta = np.frombuffer(self.eta.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        eta[:] = np.random.rand(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
The following python classes for the optimization using  a Genetic Algorithm
will be primarly be a wrapper for the pyGAD library.
"""

import pygad # type: ignore
from utils.ga_utils import *

@dataclass
class GAParameters:
    """
    Dataclass representing the parameters of the optimization algorithm.

    The parameters of the optimization algorithm:
            - n_generations : int
                The number of generations of the algorithm.
            - n_parents_mating : int
                The number of solutions to be selected as parents for the next generation
            - sol_per_pop : int
                The number of solutions in the population.
            - parent_selection_type : str
                The type of parent selection.
            - num_genes:
                The number of genes in the chromosome.
            - init_range_low : float
                The lower bound of the initial range of the solutions.
            - init_range_high : float
                The upper bound of the initial range of the solutions.
            - keep_parents : int
                The number of parents to keep in the next generation.
            - gene_type : type = int
                The type of the genes of the solutions.
            - mutation_probability : float
                The probability of mutation
            - crossover_probability : float
                The probability of crossover

    """

    n_generations : int = 100
    n_parents_mating : int = 20
    sol_per_pop : int = 20
    parent_selection_type : str = "rws"
    num_genes : int = None
    init_range_low : float = None
    init_range_high : float = None
    keep_parents : int = 1
    gene_type : type = int
    mutation_probability : float = 0.2
    crossover_probability : float = 0.8
    k_tournament : int = 3


class GeneticAlgorithm(BaseOpt):

    def __init__(self, optimization_parameters, domain, task_graph, seed = None):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__()
        self.par = optimization_parameters

        if self.par.num_genes is None:
            # the default value for the number of genes is the number of tasks in the graph
            self.par.num_genes = task_graph.n_nodes

        if self.par.init_range_low is None:
            # the default value for the lower bound of the initial range is 0
            self.par.init_range_low = 0

        if self.par.init_range_high is None:
            # the default value for the upper bound of the initial range is the size of the domain
            self.par.init_range_high = domain.size

        # --- Domain and Task Graph ---
        self.domain = domain
        self.task_graph = task_graph

        self.tasks = [task["id"] for task in self.task_graph.get_nodes()]
        
        # --- Pool of Operatros ---
        self.pool = OperatorPool(self)
        
        # --- Seed ---
        self.seed = seed
        
        # --- Gene space with source and drain nodes fixed ---
        # The gene space is a list of None values, except for the first and last genes
        self.source_node = self.task_graph.SOURCE_POINT
        self.drain_node = self.task_graph.DRAIN_POINT
        
        self.upper_latency_bound = 10000.0 #used to normalize the output of the fitness function, value is adjusted in on_start_fitness_norm() method
        
        # Create gene space with one entry per task
        self.gene_space = self.pool.create_gene_space()
        
        # --- Initialize the GA object of pyGAD ---
        self.ga_instance = pygad.GA(num_generations = self.par.n_generations,       #number of generations
                                    num_parents_mating = self.par.n_parents_mating, #Number of solutions to be selected as parents
                                    fitness_func = self.fitness_func,               #maximisation function, can be multiobjective
                                    fitness_batch_size = None,                      #you can calculate the fitness in batches, but we do not use it here (maybe for GPU)
                                    initial_population = None,                      #initial population, if None it will be generated randomly
                                    sol_per_pop = self.par.sol_per_pop,             #number of solutions in the population (chromosomes)
                                    num_genes = self.par.num_genes,                 #number of genes in the chromosome   
                                    gene_type = self.par.gene_type,                 #int
                                    init_range_low = self.par.init_range_low,       #from 0 node
                                    init_range_high = self.par.init_range_high,     #to the size of the grid
                                    parent_selection_type = self.par.parent_selection_type, #can be: Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
                                    keep_parents = self.par.keep_parents,           # Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keeps the specified number of parents in the next population. 
                                    keep_elitism = 0,                               # It can take the value 0 or a positive integer that satisfies (0 <= keep_elitism <= sol_per_pop). It defaults to 1 which means only the best solution in the current generation is kept in the next generation. If assigned 0, this means it has no effect. If assigned a positive integer K, then the best K solutions are kept in the next generation. It cannot be assigned a value greater than the value assigned to the sol_per_pop parameter. If this parameter has a value different than 0, then the keep_parents parameter will have no effect.      
                                    K_tournament = self.par.k_tournament,           # In case that the parent selection type is tournament, the K_tournament specifies the number of parents participating in the tournament selection. It defaults to 3.
                                    crossover_type = self.pool.get_cross_func,      # user-defined function: choose across possible crossovers 
                                    mutation_type = self.pool.get_mut_func,         # user-defined function: choose across possible mutations
                                    mutation_probability = self.par.mutation_probability, # probalbiity of mutation
                                    crossover_probability = self.par.crossover_probability, # probability of crossover 
                                    gene_space = self.gene_space,                   # create a space for each gene, source and drain node fixed here
                                    gene_constraint = None, #self.pool.create_gene_constraints(),                         # WARNING: Works from PyGAD 3.5.0 A list of callables (i.e. functions) acting as constraints for the gene values. Before selecting a value for a gene, the callable is called to ensure the candidate value is valid. We check here memory constraint for PEs and source and drain nodes are fixed.
                                    sample_size = 500,                              # if gene_constraint used then sample_size defines number of tries to create a gene which fulfills the constraints
                                    on_start = self.on_start_fitness_norm,                                # functiion to be called at the start of the optimization
                                    on_fitness = None,                              # function to be called after each fitness evaluation
                                    on_generation = self.pool.on_generation,        # on each generation reward is updated and different operator is picked
                                    on_stop = self.pool.on_stop,                    # save the data at the end of the optimization
                                    stop_criteria="saturate_150",                             # stop criteria for the optimization: Some criteria to stop the evolution. Added in PyGAD 2.15.0. Each criterion is passed as str which has a stop word. The current 2 supported words are reach and saturate. reach stops the run() method if the fitness value is equal to or greater than a given fitness value. An example for reach is "reach_40" which stops the evolution if the fitness is >= 40. saturate means stop the evolution if the fitness saturates for a given number of consecutive generations. An example for saturate is "saturate_7" which means stop the run() method if the fitness does not change for 7 consecutive generations.
                                    random_seed = self.seed
        )
        
        self.GA_DIR = get_GA_DIR()
        self.CONFIG_DUMP_DIR = get_CONFIG_DUMP_DIR()
        self.ARCH_FILE = get_ARCH_FILE()
        
    def on_start_fitness_norm(self, ga_instance):
        '''
        Method to be called before starting the evolution process. It is used to dynamically estimate the parameters which scales the fitness fucntion.
        '''
        init_genes = []
        # randomly assign the tasks to the PEs in the NoC
        for task in self.tasks:
            if task == 'start':
                # 'start' task must be mapped to source node
                init_genes.append(self.task_graph.SOURCE_POINT)
            elif task == 'end':
                # 'end' task must be mapped to drain node  
                init_genes.append(self.task_graph.DRAIN_POINT)
            else:
                # Regular tasks can be mapped to any PE
                init_genes.append(np.random.choice(range(self.domain.size)))
        
        mapping_norm = {}
        for task_idx, task in enumerate(self.tasks):
            if task_idx != "start" and task_idx != "end":
                mapping_norm[task] = int(init_genes[task_idx])

        # 2. apply the mapping to the task graph
        mapper_norm = ma.Mapper()
        mapper_norm.init(self.task_graph, self.domain)
        mapper_norm.set_mapping(mapping_norm)
        mapper_norm.mapping_to_json(self.CONFIG_DUMP_DIR + "/dump_GA_x.json", file_to_append=self.ARCH_FILE)

        # 3. run the simulation
        stub = ss.SimulatorStub()
        result_to_norm, _ = stub.run_simulation(self.CONFIG_DUMP_DIR + "/dump_GA_x.json", dwrap=True)
        
        print(f"Initial fitness norm result: {result_to_norm}")
        
        #result rounded up to the nearest 1000
        result_norm = int(np.ceil(result_to_norm / 1000.0)) * 1000
        
        print(f"Initial fitness norm result rounded: {result_norm}")
        
        self.upper_latency_bound = result_norm 
        
        print("\nFinished on_start_fitness_norm, upper_latency_bound set to:", self.upper_latency_bound)  
        

    def fitness_func(self, ga_instance, solution, solution_idx):

        # fitness function is computed using the NoC simulator:
        # 1. construct the mapping from the solution
        # constuct the mapping form the path
        verbose = False
        
        mapping = {}
        for task_idx, task in enumerate(self.tasks):
            if task_idx != "start" and task_idx != "end":
                mapping[task] = int(solution[task_idx])

        if not self.CONFIG_DUMP_DIR:
            raise RuntimeError("Config Dump dir not initialized. Call initialize_globals() first.")
        
        # 2. apply the mapping to the task graph
        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(self.CONFIG_DUMP_DIR + "/dump_GA_"+ str(solution_idx) + ".json", file_to_append=self.ARCH_FILE)

        # 3. run the simulation
        stub = ss.SimulatorStub()
        result, _ = stub.run_simulation(self.CONFIG_DUMP_DIR + "/dump_GA_"+ str(solution_idx) +".json", dwrap=True)
        
        if not hasattr(self, 'upper_latency_bound'):
            raise RuntimeError("upper_latency_bound not initialized. Call on_start_fitness_norm() first.")

        norm_result = self.upper_latency_bound / (result + 1e-6)
        
        if verbose:
            print(f"Solution {solution_idx} fitness: {norm_result} (raw result: {result})")
        
        return norm_result
    
    def run(self):

        self.ga_instance.run()
        return self.ga_instance.best_solution()
    
    def summary(self):
        return self.ga_instance.summary()
    
    def plot_summary(self):
        self.ga_instance.plot_fitness() #Shows how the fitness evolves by generation.
        
        #only if save solutions is set to True
        #self.ga_instance.plot_genes() #Shows how the gene value changes for each generation.
        #self.ga_instance.plot_new_solution_rate() #Shows the number of new solutions explored in each solution.
    

class ParallelGA(GeneticAlgorithm):

    def __init__(self, n_procs, optimization_parameters, domain, task_graph, seed):
        super().__init__(optimization_parameters, domain, task_graph, seed)

        self.ga_instance = pygad.GA(num_generations = self.par.n_generations,       #number of generations
                                    num_parents_mating = self.par.n_parents_mating, #Number of solutions to be selected as parents
                                    fitness_func = self.fitness_func,               #maximisation function, can be multiobjective
                                    fitness_batch_size = None,                      #you can calculate the fitness in batches, but we do not use it here (maybe for GPU)
                                    initial_population = None,                      #initial population, if None it will be generated randomly
                                    sol_per_pop = self.par.sol_per_pop,             #number of solutions in the population (chromosomes)
                                    num_genes = self.par.num_genes,                 #number of genes in the chromosome   
                                    gene_type = self.par.gene_type,                 #int
                                    init_range_low = self.par.init_range_low,       #from 0 node
                                    init_range_high = self.par.init_range_high,     #to the size of the grid
                                    parent_selection_type = self.par.parent_selection_type, #can be: Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
                                    keep_parents = self.par.keep_parents,           # Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keeps the specified number of parents in the next population. 
                                    keep_elitism = 0,                               # It can take the value 0 or a positive integer that satisfies (0 <= keep_elitism <= sol_per_pop). It defaults to 1 which means only the best solution in the current generation is kept in the next generation. If assigned 0, this means it has no effect. If assigned a positive integer K, then the best K solutions are kept in the next generation. It cannot be assigned a value greater than the value assigned to the sol_per_pop parameter. If this parameter has a value different than 0, then the keep_parents parameter will have no effect.      
                                    K_tournament = self.par.k_tournament,           # In case that the parent selection type is tournament, the K_tournament specifies the number of parents participating in the tournament selection. It defaults to 3.
                                    crossover_type = self.pool.get_cross_func,      # user-defined function: choose across possible crossovers 
                                    mutation_type = self.pool.get_mut_func,         # user-defined function: choose across possible mutations
                                    mutation_probability = self.par.mutation_probability, # probalbiity of mutation
                                    crossover_probability = self.par.crossover_probability, # probability of crossover 
                                    gene_space = self.gene_space,                   # create a space for each gene, source and drain node fixed here
                                    gene_constraint = None, #self.pool.create_gene_constraints(),                         # WARNING: Works from PyGAD 3.5.0 A list of callables (i.e. functions) acting as constraints for the gene values. Before selecting a value for a gene, the callable is called to ensure the candidate value is valid. We check here memory constraint for PEs and source and drain nodes are fixed.
                                    sample_size = 500,                              # if gene_constraint used then sample_size defines number of tries to create a gene which fulfills the constraints
                                    on_start = self.on_start_fitness_norm_parallel,                                # functiion to be called at the start of the optimization
                                    on_fitness = None,                              # function to be called after each fitness evaluation
                                    on_generation = self.pool.on_generation,        # on each generation reward is updated and different operator is picked
                                    on_stop = self.pool.on_stop,                    # save the data at the end of the optimization
                                    stop_criteria="saturate_150",                             # stop criteria for the optimization: Some criteria to stop the evolution. Added in PyGAD 2.15.0. Each criterion is passed as str which has a stop word. The current 2 supported words are reach and saturate. reach stops the run() method if the fitness value is equal to or greater than a given fitness value. An example for reach is "reach_40" which stops the evolution if the fitness is >= 40. saturate means stop the evolution if the fitness saturates for a given number of consecutive generations. An example for saturate is "saturate_7" which means stop the run() method if the fitness does not change for 7 consecutive generations.
                                    random_seed = self.seed,
                                    parallel_processing=["process", n_procs] # Use multiprocessing with n_procs processes
        )
                
    def run_norm_simulation(self, args):
        seed, task_graph, domain, tasks, config_dir, arch_file = args

        np.random.seed(seed)

        init_genes = []
        for task in tasks:
            if task == 'start':
                init_genes.append(task_graph.SOURCE_POINT)
            elif task == 'end':
                init_genes.append(task_graph.DRAIN_POINT)
            else:
                init_genes.append(np.random.choice(range(domain.size)))

        mapping_norm = {}
        for task_idx, task in enumerate(tasks):
            if task != "start" and task != "end":
                mapping_norm[task] = int(init_genes[task_idx])

        mapper_norm = ma.Mapper()
        mapper_norm.init(task_graph, domain)
        mapper_norm.set_mapping(mapping_norm)

        json_path = config_dir + f"/dump_GA_x_{seed}.json"
        mapper_norm.mapping_to_json(json_path, file_to_append=arch_file)

        stub = ss.SimulatorStub()
        result_to_norm, _ = stub.run_simulation(json_path, dwrap=True)

        return result_to_norm

        
    def on_start_fitness_norm_parallel(self, ga_instance):
        """
        Runs n_procs parallel simulations to estimate the normalization parameter for fitness.
        """

        num_jobs = ga_instance.parallel_processing[1]

        # Prepare arguments for each process
        arg_list = [
            (i, self.task_graph, self.domain, self.tasks, self.CONFIG_DUMP_DIR, self.ARCH_FILE)
            for i in range(num_jobs)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            results = list(executor.map(self.run_norm_simulation, arg_list))

        print(f"Initial fitness norm results: {results}")

        # Use the median as the normalization value, rounded up to nearest 1000
        result_norm_median = int(np.ceil(np.median(results) / 1000.0)) * 1000
        print(f"Initial fitness norm result (median, rounded): {result_norm_median}")

        #result_norm_mean = int(np.ceil(np.mean(results) / 1000.0)) * 1000
        #print(f"Initial fitness norm result (mean, rounded): {result_norm_mean}")
        
        self.upper_latency_bound = result_norm_median
        print("\nFinished on_start_fitness_norm_parallel, upper_latency_bound set to:", self.upper_latency_bound)
        

    def fitness_func(self, ga_instance, solution, solution_idx):

        # fitness function is computed using the NoC simulator:
        # 1. construct the mapping from the solution
        mapping = {}
        for task_idx, task in enumerate(self.tasks):
            if task_idx != "start" and task_idx != "end":
                mapping[task] = int(solution[task_idx])

        # 2. apply the mapping to the task graph
        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        
        if not self.CONFIG_DUMP_DIR:
            raise RuntimeError("Config Dump dir not initialized. Call initialize_globals() first.")

        # 3. determine which process is running the simulation
        mapper.mapping_to_json(self.CONFIG_DUMP_DIR + "/dump_GA_"+ str(solution_idx)+".json", file_to_append=self.ARCH_FILE)

        # 3. run the simulation
        stub = ss.SimulatorStub()
        result, _ = stub.run_simulation(self.CONFIG_DUMP_DIR + "/dump_GA_"+ str(solution_idx)+".json")
        
        norm_result = self.upper_latency_bound / (result + 1e-6)

        return norm_result