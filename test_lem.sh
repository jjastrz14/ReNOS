#!/bin/bash
#SBATCH --partition lem-cpu    
#SBATCH --job-name=renos_lem
#SBATCH --time=24:05:00       #time limit HH:MM:SS
#SBATCH --nodes=1             #reserve nodes
#SBATCH --ntasks-per-node=1              #task per all nodes
#SBATCH --cpus-per-task=128  #number of threads per task
#SBATCH --mem=150gb               #memory per node
#SBATCH --gres=storage:lustre:1
#SBATCH --mail-user=jakub.jastrzebski99@gmail.com
#SBATCH --mail-type=ALL

source /usr/local/sbin/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
module load pybind11/2.11.1-GCCcore-12.3.0

source ~/renos/bin/activate


ALGO="ACO_parallel"
RESULT_DIR="ACO_ISCAS_ResNetMid"
RESULT_DIR_HOME="/home/jjastrz9/tmp/ReNOS/data"

TMPDIR_LUSTRE="/lustre/tmp/slurm/${SLURM_JOB_ID}"
export TMPDIR_LUSTRE

# Create output directory
mkdir -p $TMPDIR_LUSTRE || { echo "tmpdir lustre not created"; exit 1; }

echo "Coping data to scratch"

for pattern in \
  build\ \
  config_files\ \
  src\ 
do
  cp -r  $pattern "$TMPDIR_LUSTRE/" || echo "Warning: failed to copy $pattern"
done

cd $TMPDIR_LUSTRE || { echo "Failed to change directory"; exit 1; }
mkdir -p data || { echo "Data dir in lustre not created"; exit 1; }

echo "Setting up job in $TMPDIR_LUSTRE ended"

# Run the simulation
echo "Starting simulation at $(date)"

python3 src/main.py -algo $ALGO -name $RESULT_DIR
#python3 src/partitioner.py

echo "Simulation completed at $(date)"

# Copy results back
echo "Copying results to $RESULT_DIR_HOME"

if ls ./data/"${ALGO}_${RESULT_DIR}_"* 1> /dev/null 2>&1; then
    cp -r "./data/${ALGO}_${RESULT_DIR}_"* "$RESULT_DIR_HOME/" || { echo "Failed to copy results"; exit 1; }
    echo "Results copied successfully at $(date)"
else
    echo "WARNING: No result files found to copy"
    echo "Files in current directory:"
    ls -la
fi

echo "Job completed: $(date)"