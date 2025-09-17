'''
==================================================
File: analytical_simulator_stub.py
Project: ReNOS
File Created: Tuesday, 17th September 2024
Author: Jakub Jastrzebski 
==================================================
'''

"""
The analytical_simulator_stub.py module contains the classes used to interface with the analytical NoC simulator
to estimate latency for a given mapping of the tasks onto the NoC grid.

This module provides the same interface as simulator_stub.py but uses the fast analytical model
instead of the cycle-accurate BookSim2 simulator.

The analytical simulator requires the same configuration (.JSON) file format as BookSim2, containing:
- the architectural parameters of the NoC grid (e.g., the number of PEs, the topology of the grid, etc.);
- the workload and corresponding mapping of the tasks onto the NoC grid.
"""

import os
import sys
import platform
import multiprocessing as mpr
from concurrent.futures import ThreadPoolExecutor
import time
from dirs import PYTHON_MODULE_DIR

# import the analytical simulator module
sys.path.append(PYTHON_MODULE_DIR)

try:
    import analytical_nocsim  # type: ignore
except ImportError:
    print("Warning: analytical_nocsim module not found. Make sure the analytical model is compiled with pybind11.")
    analytical_nocsim = None

def wrapper(func, q, event, *args):
    q.put(func(*args))
    event.set()

def dangerwrap(func, *args):
    event = mpr.Event()
    q = mpr.Queue()

    f_process = mpr.Process(target=wrapper, args=(func, q, event, *args))
    f_process.start()
    try:
        event.wait()
    except KeyboardInterrupt:
        f_process.terminate()
        f_process.join()
        print("Caught in dangerwrap")
        return None

    return q.get()

def run_analytical_simulation(path_to_config_file):
    """Run analytical simulation with the given config file"""
    if analytical_nocsim is None:
        raise RuntimeError("Analytical simulator module not available")
    return analytical_nocsim.simulate_analytical(path_to_config_file, "")


class AnalyticalSimulatorStub:
    """
    Interface to the analytical NoC simulator for fast latency estimation.
    Provides the same interface as SimulatorStub but uses the analytical model.
    """

    def __init__(self, path_to_executable=None):
        """
        Initialize the analytical simulator stub.

        Parameters
        ----------
        path_to_executable : str, optional
            Not used for analytical simulator (for compatibility with SimulatorStub)
        """
        self.path_to_executable = path_to_executable
        if analytical_nocsim is None:
            raise RuntimeError("Analytical simulator module not available. Please compile the analytical model.")

    def run_simulation(self, path_to_config_file, verbose=False, dwrap=False):
        """
        Runs the analytical simulation with the given configuration file.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.
        verbose : bool, optional
            If True, prints verbose output.
        dwrap : bool, optional
            If True, uses multiprocessing wrapper (for compatibility).

        Returns
        -------
        tuple
            (simulation_time, logger) - Results from analytical simulation
        """

        if verbose:
            print("Running analytical simulation with configuration file: " + path_to_config_file)

        start = time.time()
        # Run analytical simulation (much faster than BookSim2)
        if dwrap:
            results, logger = dangerwrap(run_analytical_simulation, path_to_config_file)
        else:
            results, logger = run_analytical_simulation(path_to_config_file)
        end = time.time()

        if verbose:
            print(f"Analytical simulation completed in {end - start:.4f} seconds.")
            print(f"Simulation time: {results} cycles")

        return results, logger

    def run_simulation_on_processor(self, path_to_config_file, processor, verbose=False):
        """
        Runs the analytical simulation with the given configuration file.
        Processor affinity setting is ignored for analytical simulation.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.
        processor : int
            The processor (ignored for analytical simulation).
        verbose : bool, optional
            If True, prints verbose output.

        Returns
        -------
        tuple
            (simulation_time, logger) - Results from analytical simulation
        """

        if verbose:
            print(f"Running analytical simulation with configuration file: {path_to_config_file}")
            print("Note: Processor affinity is not applicable for analytical simulation.")

        start = time.time()
        results, logger = run_analytical_simulation(path_to_config_file)
        end = time.time()

        if verbose:
            print(f"Analytical simulation completed in {end - start:.4f} seconds.")
            print(f"Simulation time: {results} cycles")

        return results, logger

    def run_simulations_in_parallel(self, config_files, processors=None, verbose=False):
        """
        Runs analytical simulations in parallel.

        Parameters
        ----------
        config_files : list of str
            List of paths to configuration files.
        processors : list of str, optional
            List of processors (for compatibility, not used).
        verbose : bool, optional
            If True, prints verbose output.

        Returns
        -------
        tuple
            (results, loggers) - Lists of simulation times and loggers
        """
        if verbose:
            print("Running batch of analytical simulations in parallel.")

        start = time.time()

        # Use all available CPUs if processors not specified
        if processors is None:
            max_workers = os.cpu_count()
        else:
            max_workers = len(processors)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for config_file in config_files:
                futures.append(executor.submit(self.run_simulation, config_file, False, False))
            results_loggers = [future.result() for future in futures]  # Wait for all futures to complete

        end = time.time()
        if verbose:
            print(f"Analytical simulations completed in {end - start:.4f} seconds.")

        # Separate results and loggers
        results = [result for result, logger in results_loggers]
        loggers = [logger for result, logger in results_loggers]

        return results, loggers

    def get_model_info(self):
        """
        Get information about the analytical model.

        Returns
        -------
        str
            Model information string
        """
        if analytical_nocsim is not None:
            return f"Analytical NoC Simulator v{analytical_nocsim.__version__}: {analytical_nocsim.__description__}"
        else:
            return "Analytical simulator not available"

    def validate_config(self, path_to_config_file):
        """
        Validate configuration file for analytical simulation.

        Parameters
        ----------
        path_to_config_file : str
            Path to configuration file

        Returns
        -------
        bool
            True if configuration is valid
        """
        if not os.path.exists(path_to_config_file):
            return False

        # Could add more validation here using the analytical simulator
        return True


# Factory function to create the appropriate simulator stub
def create_simulator_stub(use_analytical=True, **kwargs):
    """
    Factory function to create either analytical or cycle-accurate simulator stub.

    Parameters
    ----------
    use_analytical : bool, optional
        If True, creates AnalyticalSimulatorStub, otherwise SimulatorStub
    **kwargs
        Additional arguments passed to the simulator stub constructor

    Returns
    -------
    AnalyticalSimulatorStub or SimulatorStub
        The appropriate simulator stub
    """
    if use_analytical:
        return AnalyticalSimulatorStub(**kwargs)
    else:
        # Import the original simulator stub
        from simulator_stub import SimulatorStub
        return SimulatorStub(**kwargs)