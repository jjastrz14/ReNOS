'''
==================================================
File: partitioner_simple.py
Project: ReNOS
File Created: Wednesday, 15th October 2025
Author: Jakub Jastrzebski (jakubandrzej.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
Simple partitioner that analyzes each layer individually with different partitioning strategies.
No simulation, just statistics collection.
"""

from models import *
from utils.partitioner_utils import build_partitions_single_layer, get_valid_partition_ranges
from utils.model_fusion import fuse_conv_bn
import csv
import os
import larq


if __name__ == '__main__':

    # Model configuration
    model = ResNet_block_smaller(input_shape=(56, 56, 32), num_classes=10, verbose=True)
    model_name = 'ResNet_block'

    # Fuse Conv+BN layers
    model = fuse_conv_bn(model, verbose=True)
    
    larq.models.summary(model, print_fn=None, include_macs=True)

    # Output directory
    data_path = "./data/partitioner_data15Oct"
    os.makedirs(data_path, exist_ok=True)

    # CSV fieldnames for statistics
    # Note: All size fields are in BYTES (already multiplied by PRECISION=2 for float16)
    fieldnames = [
        'layer_index',
        'layer_name',
        'layer_type',
        'partitioning_strategy',
        'spatial',
        'out_ch',
        'in_ch',
        'num_partitions',
        'mean_partition_size',
        'min_partition_size',
        'max_partition_size',
        'std_partition_size',
        'total_size',
        'mean_partition_size_input',
        'min_partition_size_input',
        'max_partition_size_input',
        'std_partition_size_input',
        'mean_partition_size_output',
        'min_partition_size_output',
        'max_partition_size_output',
        'std_partition_size_output',
        'mean_partition_size_weights',
        'min_partition_size_weights',
        'max_partition_size_weights',
        'std_partition_size_weights',
        'mean_flops',
        'min_flops',
        'max_flops',
        'std_flops',
        'total_flops',
        'mean_macs',
        'total_macs',
        'total_weights',
        'mean_input_tensor_size',
        'min_input_tensor_size',
        'max_input_tensor_size',
        'std_input_tensor_size',
        'total_input_data',
        'number_of_input_connections',
        'mean_output_tensor_size',
        'min_output_tensor_size',
        'max_output_tensor_size',
        'std_output_tensor_size',
        'total_output_data',
        'number_of_output_connections'
    ]

    # Process each layer
    for layer_idx, layer in enumerate(model.layers):

        # Skip input and non-computational layers
        skip_layers = (layers.InputLayer, layers.Flatten, layers.Reshape,
                        layers.Add, layers.Identity)

        if isinstance(layer, skip_layers):
            print(f"Skipping layer {layer_idx}: {layer.name} ({layer.__class__.__name__})")
            continue

        print(f"\n{'='*80}")
        print(f"Processing Layer {layer_idx}: {layer.name} ({layer.__class__.__name__})")
        print(f"{'='*80}")

        # Get valid partition ranges for this layer
        ranges = get_valid_partition_ranges(layer, min_partition_size = 4)
        print(f"  Layer shapes: input={ranges.get('input_shape')}, output={ranges.get('output_shape')}")
        print(f"  Max partitions: spatial={ranges['spatial_max']} (2^n splits), "
                f"output_ch={ranges['output_ch_max']}, input_ch={ranges['input_ch_max']}")
        

        # Generate valid combinations for this layer
        # spatial: 0 to spatial_max (represents powers of 2: 0->1, 1->2, 2->4, etc.)
        # output_ch: 1 to output_ch_max
        # input_ch: 1 to input_ch_max
        combinations = [
            (i, j, k)
            for i in range(0, ranges['spatial_max'] + 1)
            for j in range(1, ranges['output_ch_max'] + 1) 
            for k in range(1, ranges['input_ch_max'] + 1)  
        ]

        print(f"  Total combinations to test: {len(combinations)}\n")
                
        # Create separate CSV file for this layer
        layer_csv = f"{data_path}/layer_{layer_idx}_{layer.name}.csv"

        with open(layer_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Test all partition combinations for this layer
            for combo_idx, (i, j, k) in enumerate(combinations, 1):
                print(f"  [{combo_idx}/{len(combinations)}] Testing partition: spatial={i}, out_ch={j}, in_ch={k}")

                try:
                    # Analyze this layer with this partitioning strategy
                    partitions, stats = build_partitions_single_layer(
                        layer,
                        partitioning_tuple=(i, j, k),
                        verbose=False
                    )

                    # Add partitioning config to stats
                    stats['layer_index'] = layer_idx
                    stats['spatial'] = i
                    stats['out_ch'] = j
                    stats['in_ch'] = k

                    # Write to CSV
                    writer.writerow(stats)

                    print(f"    ✓ Partitions: {stats['num_partitions']}, "
                            f"Total FLOPs: {stats.get('total_flops', 0):,}, "
                            f"Total size: {stats.get('total_size', 0):,} bytes")

                except Exception as e:
                    print(f"    ✗ ERROR: {str(e)}")
                    # Write error entry
                    writer.writerow({
                        'layer_index': layer_idx,
                        'layer_name': layer.name,
                        'layer_type': layer.__class__.__name__,
                        'spatial': i,
                        'out_ch': j,
                        'in_ch': k,
                        'num_partitions': 'ERROR',
                    })

        print(f"\n  Results saved to: {layer_csv}")

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Per-layer CSV files saved to: {data_path}/")
    print(f"{'='*80}")
