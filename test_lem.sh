#!/bin/bash
#SBATCH --partition lem-cpu    
#SBATCH --job-name=renos_lem
#SBATCH --time=05:05:00         #time limit
#SBATCH --nodes=1             #reserve nodes
#SBATCH --ntasks-per-node=1              #task per all nodes
#SBATCH --cpus-per-task=128   #number of threads per task
#SBATCH --mem=25gb               #memory per node
#SBATCH --gres=storage:lustre:1
#SBATCH --mail-user=jakub.jastrzebski99@gmail.com
#SBATCH --mail-type=ALL

source ~/renos/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load pybind11/2.11.1-GCCcore-12.3.0

RESULT_DIR="LeNet4_run"

# Run the simulation
echo "Starting simulation at $(date)"

python3 src/main.py -algo ACO -name $RESULT_DIR

echo "Simulation completed at $(date)"

echo "Job completed: $(date)"