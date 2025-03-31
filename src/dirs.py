import os
import shutil
import time
import json
import pytz
from datetime import datetime

def get_json_field(json_file_path, field_name, default_value=None):
    """
    Extract a specific field from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        field_name (str): Name of the field to extract
        default_value: Value to return if the field doesn't exist
        
    Returns:
        The value of the specified field, or default_value if not found
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
            # For fields inside arch object without specifying arch prefix
            if field_name in data.get('arch', {}):
                return data['arch'][field_name]
        
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"Error reading JSON field '{field_name}' from {json_file_path}: {e}")
        return default_value

####################################################################################################
# Directories
####################################################################################################
    

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SRC_DIR)
CONFIG_FILES_DIR = os.path.join(MAIN_DIR, "config_files")
RESTART_DIR = os.path.join(SRC_DIR, "restart")
RUN_FILES_DIR = os.path.join(CONFIG_FILES_DIR, "runs")
DATA_DIR = os.path.join(MAIN_DIR, "data")
PYTHON_MODULE_DIR = os.path.join(MAIN_DIR, "build/lib")

DEFALUT_ARCH_FILE = os.path.join(CONFIG_FILES_DIR, "arch.json")
DEFALUT_CONFIG_DUMP_DIR = os.path.join(CONFIG_FILES_DIR, "dumps")

comp_cycles = get_json_field(DEFALUT_ARCH_FILE, "ANY_comp_cycles", 0)

if comp_cycles == 0.25:
    factor = "x1"
elif comp_cycles == 0.125:
    factor = "x2"
elif comp_cycles == 0.05:
    factor = "x5"
elif comp_cycles == 0.025:
    factor = "x10"
elif comp_cycles == 0.1:
    factor = "x25"
else:
    factor = comp_cycles

#copty the default arch file to the new arch file
shutil.copy(DEFALUT_ARCH_FILE, os.path.join(CONFIG_FILES_DIR, f"arch_{factor}_.json"))
ARCH_FILE = os.path.join(CONFIG_FILES_DIR, f"arch_{factor}_.json")

cet = pytz.timezone('CET')
timestamp = datetime.now(cet).strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
CONFIG_DUMP_DIR = DEFALUT_CONFIG_DUMP_DIR +"/dumps"+ f"_{factor}" + f"_{timestamp}"
ACO_DIR = os.path.join(DATA_DIR, "ACO") + f"_{factor}" + f"_{timestamp}"
GA_DIR = os.path.join(DATA_DIR, "GA") + f"_{factor}" + f"_{timestamp}"


