#!/bin/bash
#SBATCH --partition lem-cpu    
#SBATCH --job-name=renos_test_lem
#SBATCH --time=00:15:00         #time limit
#SBATCH --nodes=1             #reserve nodes
#SBATCH --ntasks-per-node=1              #task per all nodes
#SBATCH --cpus-per-task=12     #number of threads per task
#SBATCH --mem=5gb               #memory per node
#SBATCH --gres=storage:lustre:1
#SBATCH --mail-user=jakub.jastrzebski99@gmail.com
#SBATCH --mail-type=ALL

source ~/renos/bin/activate
ml Python/3.11.3-GCCcore-12.3.0
ml pybind11/2.11.1-GCCcore-12.3.0

RESULT_DIR="test_lem"
WORK_DIR="$TMPDIR_LUSTRE/$RESULT_DIR"

# Create output directory in home (permanent storage)
mkdir -p ~/$RESULT_DIR || { echo "Result dir not created in home"; exit 1; }
# Create working directory in temporary storage
mkdir -p $TMPDIR_LUSTRE/$RESULT_DIR || { echo "Temp working dir not created"; exit 1; }
# Change to working directory
cd $WORK_DIR || { echo "Failed to change directory to $WORK_DIR"; exit 1; }

echo "Setting up job in $WORK_DIR ended"

# Run the simulation
echo "Starting simulation at $(date)"

python3 src/main.py -algo ACO -name $RESULT_DIR

echo "Simulation completed at $(date)"

# Copy ALL results from current working directory to home
echo "Copying all results to home directory: ~/$RESULT_DIR"

# Method 2: Alternative - copy everything from the result directory if it exists
if [ -d "$RESULT_DIR" ]; then
  echo "Copying entire result directory contents..."
  cp -r $RESULT_DIR/* ~/$RESULT_DIR/ || echo "Warning: failed to copy directory contents"
fi

echo "Results copied to ~/$RESULT_DIR at $(date)"

# Verify the copy was successful
echo "Files in result directory:"
ls -la ~/$RESULT_DIR/

echo "Job completed: $(date)"