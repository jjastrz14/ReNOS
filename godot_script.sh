#!/bin/bash

# Configuration parameters
MODEL_NAME="AlexNet"
CONFIG="1 2 3"
MAPPING="row_wise"
RUNS=100
RUN_BOOKSIM="True"

# Tmux session name with timestamp to ensure uniqueness
SESSION_NAME="renos_mapping_$(date +%Y%m%d_%H%M%S)"

# Create a new tmux session (detached) and run the command
tmux new-session -d -s "$SESSION_NAME" \
  "source $(conda info --base)/etc/profile.d/conda.sh && conda activate renos && cd Projects/ReNOS && export CUDA_VISIBLE_DEVICES=\"\" && python3 src/mapping.py -model $MODEL_NAME -config $CONFIG -mapping $MAPPING -runs $RUNS -run_booksim $RUN_BOOKSIM; echo 'Press any key to exit...'; read -n 1"

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
