
import numpy as np
import pandas as pd
import os 
import sys
import itertools
import time
import matplotlib.pyplot as plt


# Import plotting functions from plotting_utils
import importlib
import plotting_utils

# Reload the module to get latest changes
importlib.reload(plotting_utils)

from plotting_utils import (
    plot_noc_comparison_ieee,
    plot_latency_breakdown_new_3_bars,
    plot_latency_breakdown_with_error,
    plot_latency_breakdown_with_trend,
    plot_latency_breakdown_with_error_from_two_files_with_broken_axis,
    plot_noc_comparison_latencies_ieee,
    plot_latency_breakdown_ieee,
    plot_energy_breakdown_ieee,
    plot_max_packet_delay_ieee,
    plot_avg_packet_size_ieee,
    plot_packet_delay_and_size_combined_ieee,
    plot_parallel_computations_ieee,
    plot_latency_breakdown_4_bars
)

data_noc_comp_fixed_num_parts = "../data/noc_comp_flops_sizes_29Oct/fixed_partitions_noc_comparison_ResNet_block_smaller.csv"
data_noc_comp_fixed_flops = "../data/noc_comp_flops_sizes_29Oct/flops_noc_comparison_ResNet_block_smaller.csv"
data_noc_comp_fixed_sizes = "../data/noc_comp_flops_sizes_29Oct/sizes_noc_comparison_ResNet_block_smaller.csv"

data_energy_fixed_num_parts = "../data/noc_comp_flops_sizes_29Oct/fixed_partitions_latency_energy_ResNet_block_smaller.csv"
data_energy_fixed_flops = "../data/noc_comp_flops_sizes_29Oct/flops_latency_energy_ResNet_block_smaller.csv"
data_energy_fixed_sizes = "../data/noc_comp_flops_sizes_29Oct/sizes_latency_energy_ResNet_block_smaller.csv"

fig, ax = plot_latency_breakdown_4_bars(    
    csv_path=data_energy_fixed_num_parts,
    figsize=(5.0, 2.5),  # Wider than the previous plot
    save_path='../data/test_1_4_bars.png',
    dpi=800,
    min_x_value=30,  # Only show results for 0+ partitions per layer
    max_x_value=45,
    take_avg=False,  # Take the best (minimum latency) for duplicates
    x_flops=False,
    x_sizes=False,
    show_legend=True,
    y_scientific=True,
    y_max=None
)