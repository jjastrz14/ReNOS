#!/usr/bin/env python3
"""
Script to merge batch CSV files from NoC simulations.

This script combines multiple batch CSV files into consolidated files.
It handles three types of partitioning strategies (fixed_partitions, flops, sizes)
and two types of metrics (latency_energy, noc_comparison).
"""

import pandas as pd
import glob
import os
import re
from pathlib import Path


def merge_batch_csvs(data_dir: str, output_dir: str = None):
    """
    Merge batch CSV files into consolidated files.

    Args:
        data_dir: Directory containing the batch CSV files
        output_dir: Directory to save merged files (defaults to data_dir)
    """
    if output_dir is None:
        output_dir = data_dir

    # Define the six output file patterns
    output_patterns = [
        "fixed_partitions_latency_energy_ResNet_block_smaller.csv",
        "fixed_partitions_noc_comparison_ResNet_block_smaller.csv",
        "flops_latency_energy_ResNet_block_smaller.csv",
        "flops_noc_comparison_ResNet_block_smaller.csv",
        "sizes_latency_energy_ResNet_block_smaller.csv",
        "sizes_noc_comparison_ResNet_block_smaller.csv",
    ]

    # Store merged dataframes for later processing
    merged_dfs = {}

    # Process each output file pattern
    for output_pattern in output_patterns:
        # Extract the matching pattern for batch files
        batch_pattern = f"batch*_{output_pattern}"
        batch_files = sorted(glob.glob(os.path.join(data_dir, batch_pattern)))

        if not batch_files:
            print(f"Warning: No batch files found for pattern: {batch_pattern}")
            continue

        print(f"\nMerging {len(batch_files)} files for {output_pattern}:")
        for bf in batch_files:
            print(f"  - {os.path.basename(bf)}")

        # Read and concatenate all batch files
        dfs = []
        for batch_file in batch_files:
            try:
                df = pd.read_csv(batch_file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {batch_file}: {e}")

        if not dfs:
            print(f"Error: No valid data found for {output_pattern}")
            continue

        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_dfs[output_pattern] = merged_df

        # Save merged file
        output_path = os.path.join(output_dir, output_pattern)
        merged_df.to_csv(output_path, index=False)
        print(f"✓ Saved merged file: {output_path}")
        print(f"  Total rows: {len(merged_df)}")

    # Add target_value column from noc_comparison to latency_energy files for flops and sizes
    pairs_to_merge = [
        ("flops_latency_energy_ResNet_block_smaller.csv",
         "flops_noc_comparison_ResNet_block_smaller.csv"),
        ("sizes_latency_energy_ResNet_block_smaller.csv",
         "sizes_noc_comparison_ResNet_block_smaller.csv"),
    ]

    for latency_energy_file, noc_comparison_file in pairs_to_merge:
        if latency_energy_file not in merged_dfs or noc_comparison_file not in merged_dfs:
            print(f"\nWarning: Skipping target_value merge for {latency_energy_file} - files not found")
            continue

        print(f"\nAdding target_value column to {latency_energy_file}:")

        latency_df = merged_dfs[latency_energy_file]
        comparison_df = merged_dfs[noc_comparison_file]

        # Check if target_value exists in comparison file
        if 'target_value' not in comparison_df.columns:
            print(f"  Warning: target_value column not found in {noc_comparison_file}")
            continue

        # Verify both files have the same number of rows
        if len(latency_df) != len(comparison_df):
            print(f"  Warning: Row count mismatch! latency_energy has {len(latency_df)} rows, noc_comparison has {len(comparison_df)} rows")
            continue

        # Simply append the target_value column from comparison to latency_energy
        latency_df['target_value'] = comparison_df['target_value'].values

        # Save the enhanced file
        output_path = os.path.join(output_dir, latency_energy_file)
        latency_df.to_csv(output_path, index=False)
        print(f"  ✓ Added target_value column")
        print(f"  ✓ Updated file: {output_path}")
        print(f"  Total rows: {len(latency_df)}")


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge batch CSV files from NoC simulations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/noc_comp_flops_sizes_29Oct",
        help="Directory containing batch CSV files (default: data/noc_comp_flops_sizes_29Oct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save merged files (default: same as data-dir)"
    )

    args = parser.parse_args()

    # Resolve data directory path
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        # If relative path, resolve from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)

    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return 1

    print(f"Processing batch files in: {data_dir}")
    merge_batch_csvs(data_dir, args.output_dir)
    print("\n✓ All merging complete!")

    return 0


if __name__ == "__main__":
    exit(main())
