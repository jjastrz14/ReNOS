'''
==================================================
File: dirs.py
Project: ReNOS
File Created: Wednesday, 21th May 2025
Author: Jakub Jastrzebski and Edoardo Cabiati (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The dirs.py module contains the DirectoryManager class, which is responsible for managing the directories and files used in the project and called via mutliple processes.
"""


import os
import shutil
import json
import pytz
import sys
from typing import Optional
from datetime import datetime

# -----------------------------------------------------------------------------
# Static Directory Path)
# -----------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SRC_DIR)
CONFIG_FILES_DIR = os.path.join(MAIN_DIR, "config_files")
RESTART_DIR = os.path.join(SRC_DIR, "restart")
RUN_FILES_DIR = os.path.join(CONFIG_FILES_DIR, "runs")
DATA_DIR = os.path.join(MAIN_DIR, "data")
PYTHON_MODULE_DIR = os.path.join(MAIN_DIR, "build/lib")
DEFAULT_ARCH_FILE = os.path.join(CONFIG_FILES_DIR, "arch.json")
DEFAULT_CONFIG_DUMP_DIR = os.path.join(CONFIG_FILES_DIR, "dumps")

# Create static directories
os.makedirs(CONFIG_FILES_DIR, exist_ok=True)
os.makedirs(RESTART_DIR, exist_ok=True)
os.makedirs(RUN_FILES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PYTHON_MODULE_DIR, exist_ok=True)
os.makedirs(DEFAULT_CONFIG_DUMP_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Class-based Directory Manager
# -----------------------------------------------------------------------------

class DirectoryManager:
    def __init__(self):
        self._initialized = False
        self._shared_timestamp = None
        self.arch_file = None
        self.config_dump_dir = None
        self.aco_dir = None
        self.ga_dir = None
        self.log_file_path_ga = None
        self.log_file_path_aco = None
    
    def get_json_field(self, json_file_path, field_name, default_value=None):
        """Extract a specific field from a JSON file."""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                return data.get('arch', {}).get(field_name, default_value)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(f"Error reading JSON field '{field_name}' from {json_file_path}: {e}")
            return default_value
    
    def _calculate_factor(self, comp_cycles):
        """Determine the factor string based on comp_cycles value."""
        factors = {
            0.25: "x1",
            0.125: "x2",
            0.05: "x5",
            0.025: "x10",
            0.01: "x25"
        }
        
        if comp_cycles not in factors:
            print(f"Warning: comp_cycles value {comp_cycles} not in predefined factors, using default")
            return str(comp_cycles).replace(".", "_")
        
        return factors.get(comp_cycles)
    
    def initialize(self, input_algo, name_dir):
        """Initialize all dynamic paths and timestamps"""
        if self._initialized:
            return
        
        use_speedup_factor = False  # Set to False if you want to disable speedup factor calculation
        
        print("Starting directory initialization...")
        
        # Check if DEFAULT_ARCH_FILE exists
        if not os.path.exists(DEFAULT_ARCH_FILE):
            print(f"Warning: DEFAULT_ARCH_FILE does not exist at {DEFAULT_ARCH_FILE}")
            # Create an empty arch.json file with minimal contents
            with open(DEFAULT_ARCH_FILE, 'w') as f:
                json.dump({"arch": {"ANY_comp_cycles": 0.25}}, f)
        
        if use_speedup_factor:
            # Calculate factor
            comp_cycles = self.get_json_field(DEFAULT_ARCH_FILE, "ANY_comp_cycles", 0.25)
            factor = self._calculate_factor(comp_cycles)
            print(f"Calculated factor: {factor} from comp_cycles: {comp_cycles}")
            name_dir_to_use = name_dir + f"_{factor}"
            print(f"Updated name_dir with factor: {name_dir}")
        else:
            name_dir_to_use = name_dir
            print(f"Using name_dir: {name_dir}")
        
            
        # Generate timestamp
        cet = pytz.timezone('CET')
        self._shared_timestamp = datetime.now(cet).strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Generated timestamp: {self._shared_timestamp}")
        
        # Create arch file copy
        arch_filename = f"arch_{name_dir_to_use}_.json"
        self.arch_file = os.path.join(CONFIG_FILES_DIR, arch_filename)
        try:
            shutil.copy(DEFAULT_ARCH_FILE, self.arch_file)
            print(f"Created arch file copy at: {self.arch_file}")
        except Exception as e:
            print(f"Error copying arch file: {e}")
        
        # Create both directories
        self.aco_dir = os.path.join(DATA_DIR, f"ACO_{name_dir_to_use}_{self._shared_timestamp}")
        self.ga_dir = os.path.join(DATA_DIR, f"GA_{name_dir_to_use}_{self._shared_timestamp}")
        
        # Set CONFIG_DUMP_DIR and ACO_DIR based on algorithm
        if input_algo == "ACO" or input_algo == "ACO_parallel":
            os.makedirs(self.aco_dir, exist_ok=True)
            print(f"Created ACO_DIR: {self.aco_dir}")
            self.config_dump_dir = os.path.join(DEFAULT_CONFIG_DUMP_DIR, f"dumps_ACO_{name_dir_to_use}_{self._shared_timestamp}")
            self.log_file_path_aco = os.path.join(self.aco_dir, f"log_ACO_{self._shared_timestamp}.out")
            print(f"Log paths set - ACO: {self.log_file_path_aco}")
        elif input_algo == "GA" or input_algo == "GA_parallel":
            os.makedirs(self.ga_dir, exist_ok=True)
            print(f"Created GA_DIR: {self.ga_dir}")
            self.config_dump_dir = os.path.join(DEFAULT_CONFIG_DUMP_DIR, f"dumps_GA_{name_dir_to_use}_{self._shared_timestamp}")
            self.log_file_path_ga = os.path.join(self.ga_dir, f"log_GA_{self._shared_timestamp}.out")
            print(f"Log paths set - GA: {self.log_file_path_ga}")
        else:
            raise ValueError("Invalid algorithm specified. Use 'ACO' or 'GA'.")
        
        # Create config dump directory
        os.makedirs(self.config_dump_dir, exist_ok=True)
        print(f"Created CONFIG_DUMP_DIR: {self.config_dump_dir}")
        
        
        self._initialized = True
        print("Directory initialization complete.")
    
    def get_timestamp(self):
        if not self._initialized:
            raise RuntimeError("Directory manager not initialized. Call initialize() first.")
        return self._shared_timestamp
    
    def get_aco_log_path(self):
        if not self._initialized:
            raise RuntimeError("Directory manager not initialized. Call initialize() first.")
        return self.log_file_path_aco
    
    def get_ga_log_path(self):
        if not self._initialized:
            raise RuntimeError("Directory manager not initialized. Call initialize() first.")
        return self.log_file_path_ga
    
    def debug_print(self):
        """Print current variable values for debugging."""
        print("\nDebug Directory Manager:")
        print(f"_initialized: {self._initialized}")
        print(f"_shared_timestamp: {self._shared_timestamp}")
        print(f"aco_dir: {self.aco_dir}")
        print(f"ga_dir: {self.ga_dir}")
        print(f"arch_file: {self.arch_file}")
        print(f"config_dump_dir: {self.config_dump_dir}")
        print(f"log_file_path_ga: {self.log_file_path_ga}")
        print(f"log_file_path_aco: {self.log_file_path_aco}")

# Create a singleton instance
_dir_manager = DirectoryManager()

# -----------------------------------------------------------------------------
# Public API (functions and variables to be imported)
# -----------------------------------------------------------------------------

# Functions
def initialize_globals(input_algo, name_dir):
    _dir_manager.initialize(input_algo, name_dir)

def get_timestamp():
    return _dir_manager.get_timestamp()

def get_aco_log_path():
    return _dir_manager.get_aco_log_path()

def get_ga_log_path():
    return _dir_manager.get_ga_log_path()

def debug_globals():
    _dir_manager.debug_print()

# Global variables that can be imported
def get_ACO_DIR():
    return _dir_manager.aco_dir

def get_GA_DIR():
    return _dir_manager.ga_dir

def get_ARCH_FILE():
    return _dir_manager.arch_file

def get_CONFIG_DUMP_DIR():
    return _dir_manager.config_dump_dir

# For testing
if __name__ == "__main__":
    print("Testing dirs module...")
    try:
        initialize_globals("ACO")
        debug_globals()
        
        # Update the global variables to match the manager's state
        ACO_DIR = get_ACO_DIR()
        GA_DIR = get_GA_DIR()
        ARCH_FILE = get_ARCH_FILE()
        CONFIG_DUMP_DIR = get_CONFIG_DUMP_DIR()
        
        print(f"ACO_DIR: {ACO_DIR}")
        print(f"GA_DIR: {GA_DIR}")
    except Exception as e:
        print(f"Error during testing: {e}")