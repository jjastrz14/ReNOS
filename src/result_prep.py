'''
==================================================
File: result_prep.py
Project: ReNOS
File Created: Tuesday, 13th October 2025
Author: Jakub Jastrzebski (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

import json
import time
import csv
import simulator_stub as ss
import fast_analytical_simulator_stub as ssfam

# ============================================================================
# SET YOUR PATHS HERE
# ============================================================================
data_dir = "./data/mapping_comparison_13October"  # Directory for data files
best_sol = "ACO_VGG_late_fixed_tuple_run_row_wise_true_2025-10-11_08-35-09"  # Path to your config JSON
json_path = f"{data_dir}/{best_sol}/best_solution.json"  # Path to your config JSON
csv_path = f"{data_dir}/mapping_comparison_VGG_16_late.csv"  # Path to your CSV file
mapping_name = "ACO_row"  # Mapping strategy name

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

# Initialize simulators
fast_sim = ssfam.FastAnalyticalSimulatorStub()
booksim_stub = ss.SimulatorStub()

# 1. Run Fast Analytical model
print("\nRunning Fast Analytical model simulation...")
start_time = time.time()
result_fast_anal, logger_fast_anal = fast_sim.run_simulation(json_path, verbose=False)
fast_analytical_time = time.time() - start_time
print(f"  Result: {result_fast_anal} cycles")
print(f"  Time: {fast_analytical_time:.4f} seconds")

# 2. Enable logger and sim_power for Booksim
print("\nEnabling logger and sim_power in config...")
with open(json_path, 'r') as f:
    config = json.load(f)

if 'arch' in config:
    config['arch']['logger'] = 1
    config['arch']['sim_power'] = 1
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("  ✓ Enabled logger and sim_power")
else:
    print("  ✗ Warning: Config missing 'arch' section")

# 3. Run Booksim2 simulation
print("\nRunning Booksim2 simulation...")
start_time = time.time()
result_booksim, logger_booksim = booksim_stub.run_simulation(json_path, dwrap=False)
booksim_time = time.time() - start_time
print(f"  Result: {result_booksim} cycles")
print(f"  Time: {booksim_time:.4f} seconds")

# 4. Calculate comparison metrics
percentage_diff = abs(result_fast_anal - result_booksim) / result_booksim * 100
time_gain = booksim_time / fast_analytical_time

print(f"\nComparison:")
print(f"  Difference: {abs(result_fast_anal - result_booksim)} cycles ({percentage_diff:.2f}%)")
print(f"  Time gain: {time_gain:.4f}x")

# 5. Append to CSV
print(f"\nAppending results to {csv_path}...")

# Read the header to get fieldnames
with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames

# Create row_data with NaN for missing columns
row_data = {field: 'NaN' for field in fieldnames}  # Initialize all fields with NaN

# Fill in the data we have
row_data.update({
    'mapping_strategy': mapping_name,
    'result_analytical': result_fast_anal,
    'analytical_time': f"{fast_analytical_time:.4f}",
    'result_booksim': result_booksim,
    'booksim_time': f"{booksim_time:.4f}",
    'percentage_diff': f"{percentage_diff:.2f}",
    'time_gain': f"{time_gain:.4f}"
})

# Append the row
with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row_data)

print(f"✓ Results appended successfully!")
