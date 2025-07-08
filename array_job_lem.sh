#!/bin/bash
#SBATCH --partition lem-cpu    
#SBATCH --job-name=renos_test_lem
#SBATCH --time=05:05:00         #time limit
#SBATCH --nodes=1             #reserve nodes
#SBATCH --ntasks-per-node=1              #task per all nodes
#SBATCH --cpus-per-task=128   #number of threads per task
#SBATCH --mem=30gb               #memory per node
#SBATCH --gres=storage:lustre:1
#SBATCH --mail-user=jakub.jastrzebski99@gmail.com
#SBATCH --mail-type=ALL

# Array job specification - run jobs with indices 1-100
#SBATCH --array=1-100

source ~/renos/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load pybind11/2.11.1-GCCcore-12.3.0

# Add array task ID to result directory name
RESULT_DIR="LeNet4_run_aray_${SLURM_ARRAY_TASK_ID}"

echo "Starting array job ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "Result directory: $RESULT_DIR"

# Run the simulation
echo "Starting simulation at $(date)"

python3 src/main.py -algo ACO -name $RESULT_DIR

echo "Simulation completed at $(date)"

echo "Array job ${SLURM_ARRAY_TASK_ID} completed: $(date)"