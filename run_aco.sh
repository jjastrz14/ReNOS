#!/bin/bash
# This script runs the Ant Colony Optimization (ACO) algorithm multiple times.
# Ensure the script is executable: chmod +x run_aco.sh
# Usage: ./run_aco.sh

iteration=4

for ((i=1; i<=iteration; i++))
do
    echo "Running iteration $i"
    python3 src/main.py -algo ACO
    echo "Waiting before next iteration..."
    sleep 30
done

echo "All iterations completed."