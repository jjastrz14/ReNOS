"""
Model Fusion Utilities for NoC Simulation

This module provides functions to fuse layers (e.g., Conv + BatchNormalization)
for inference simulation purposes. The fusion is performed to simplify the
computational graph and improve partitioning efficiency.

Author: Jakub Jastrzebski
Date: 2025-10-05
"""

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from typing import Dict, List, Tuple, Optional


def fuse_conv_bn(model: keras.Model, verbose: bool = False) -> keras.Model:
    """
    Fuse Conv + BatchNormalization layer pairs for inference simulation.

    This function detects Conv → BN patterns in the model and fuses them into
    a single Conv layer with modified weights and biases. The fusion is done
    for simulation purposes to estimate FLOPs/MACs and memory footprint.

    Supported Conv types:
    - Conv2D + BatchNormalization
    - Conv1D + BatchNormalization
    - DepthwiseConv2D + BatchNormalization

    Args:
        model: Keras model to process
        verbose: If True, print fusion information

    Returns:
        New Keras model with fused Conv+BN layers

    Note:
        - Only direct Conv → BN connections are fused
        - Conv layers without BN are left unchanged
        - BN layers without preceding Conv are left unchanged
        - The fusion modifies the graph structure but uses dummy weight values
          (since we only care about shapes/sizes for simulation)
    """

    if verbose:
        print("\n" + "="*80)
        print("CONV + BN FUSION FOR INFERENCE SIMULATION")
        print("="*80)

    # Step 1: Identify fusable Conv → BN pairs
    fusable_pairs = _identify_fusable_pairs(model, verbose)

    if not fusable_pairs:
        if verbose:
            print("No fusable Conv+BN pairs found. Returning original model.")
        return model

    # Step 2: Build fused model
    fused_model = _build_fused_model(model, fusable_pairs, verbose)

    if verbose:
        print("\n" + "="*80)
        print(f"Fusion complete! Fused {len(fusable_pairs)} Conv+BN pairs.")
        fused_model.summary()
        print("="*80 + "\n")

    return fused_model


def _identify_fusable_pairs(model: keras.Model, verbose: bool = False) -> List[Tuple[str, str]]:
    """
    Identify Conv → BN layer pairs that can be fused.

    Args:
        model: Keras model to analyze
        verbose: Print debug information

    Returns:
        List of (conv_layer_name, bn_layer_name) tuples
    """

    fusable_pairs = []
    fusable_conv_types = (layers.Conv2D, layers.Conv1D, layers.DepthwiseConv2D)

    if verbose:
        print("\nScanning for fusable Conv+BN pairs...")

    # Build a map of layer connections
    layer_outputs = {}  # layer_name → list of output layer names

    for layer in model.layers:
        # Get inbound layers for this layer
        inbound_layers = []
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                inbound_layers.extend(node.inbound_layers)
            else:
                inbound_layers.append(node.inbound_layers)

        # Record connections
        for inbound in inbound_layers:
            if inbound.name not in layer_outputs:
                layer_outputs[inbound.name] = []
            layer_outputs[inbound.name].append(layer.name)

    # Find Conv → BN pairs (direct connection only)
    for layer in model.layers:
        if isinstance(layer, fusable_conv_types):
            # Check if this Conv has exactly one output
            outputs = layer_outputs.get(layer.name, [])

            if len(outputs) == 1:
                # Check if the output is a BatchNormalization layer
                output_layer_name = outputs[0]
                output_layer = model.get_layer(output_layer_name)

                if isinstance(output_layer, layers.BatchNormalization):
                    fusable_pairs.append((layer.name, output_layer.name))
                    if verbose:
                        print(f"  ✓ Found fusable pair: {layer.name} → {output_layer_name}")

    if verbose and not fusable_pairs:
        print("  No fusable pairs found.")

    return fusable_pairs


def _build_fused_model(model: keras.Model, fusable_pairs: List[Tuple[str, str]],
                       verbose: bool = False) -> keras.Model:
    """
    Build a new model with Conv+BN pairs fused.

    Args:
        model: Original model
        fusable_pairs: List of (conv_name, bn_name) tuples to fuse
        verbose: Print debug information

    Returns:
        New fused model
    """

    if verbose:
        print("\nRebuilding model with fused layers...")

    # Create a set of BN layers to skip
    bn_to_skip = {bn_name for _, bn_name in fusable_pairs}
    conv_to_fuse = {conv_name for conv_name, _ in fusable_pairs}

    # Map old layer names to new layer outputs (tensors)
    layer_map = {}

    # Rebuild model using Functional API
    # Start with input layer(s)
    if isinstance(model.input, list):
        inputs = [layers.Input(shape=inp.shape[1:], name=f"input_{i}")
                  for i, inp in enumerate(model.input)]
        for i, inp_layer in enumerate(model.layers):
            if isinstance(inp_layer, layers.InputLayer):
                layer_map[inp_layer.name] = inputs[i]
    else:
        inputs = layers.Input(shape=model.input.shape[1:], name=model.layers[0].name)
        layer_map[model.layers[0].name] = inputs

    # Process layers in topological order
    for layer in model.layers[1:]:  # Skip input layer(s)

        # Skip BN layers that are being fused
        if layer.name in bn_to_skip:
            if verbose:
                print(f"  Skipping BN layer: {layer.name} (fused into Conv)")
            continue

        # Get input tensor(s) for this layer
        inbound_tensors = _get_inbound_tensors(layer, layer_map)

        if inbound_tensors is None:
            if verbose:
                print(f"  Warning: Could not resolve inputs for layer {layer.name}, skipping")
            continue

        # If this is a Conv layer to be fused, create fused Conv
        if layer.name in conv_to_fuse:
            # Find the corresponding BN layer
            bn_name = next(bn for conv, bn in fusable_pairs if conv == layer.name)
            bn_layer = model.get_layer(bn_name)

            fused_layer, output_tensor = _create_fused_conv_layer(
                layer, bn_layer, inbound_tensors, verbose
            )

            # Map both Conv and BN to the fused output
            layer_map[layer.name] = output_tensor
            layer_map[bn_name] = output_tensor  # BN output = fused Conv output

        else:
            # Regular layer - clone and apply
            cloned_layer = _clone_layer(layer)
            output_tensor = cloned_layer(inbound_tensors)
            layer_map[layer.name] = output_tensor

    # Get output tensor(s)
    if isinstance(model.output, list):
        outputs = [layer_map[out.node.layer.name] for out in model.output]
    else:
        output_layer_name = model.output.node.layer.name
        outputs = layer_map[output_layer_name]

    # Create new model
    fused_model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_fused")

    return fused_model


def _get_inbound_tensors(layer, layer_map):
    """Get input tensor(s) for a layer from the layer_map."""

    inbound_tensors = []

    for node in layer._inbound_nodes:
        if isinstance(node.inbound_layers, list):
            for inbound_layer in node.inbound_layers:
                if inbound_layer.name in layer_map:
                    inbound_tensors.append(layer_map[inbound_layer.name])
        else:
            if node.inbound_layers.name in layer_map:
                inbound_tensors.append(layer_map[node.inbound_layers.name])

    if not inbound_tensors:
        return None

    # Return single tensor or list
    return inbound_tensors[0] if len(inbound_tensors) == 1 else inbound_tensors


def _create_fused_conv_layer(conv_layer, bn_layer, input_tensor, verbose: bool = False):
    """
    Create a fused Conv layer that combines Conv + BN.

    For simulation purposes, we:
    1. Copy Conv configuration but set use_bias=True
    2. Copy original Conv weights (dummy fusion - we only care about shapes)
    3. Create a dummy bias vector (shape = output channels)

    Args:
        conv_layer: Original Conv layer
        bn_layer: Original BN layer
        input_tensor: Input tensor for the fused layer
        verbose: Print debug info

    Returns:
        (fused_layer, output_tensor) tuple
    """

    # Get Conv layer configuration
    config = conv_layer.get_config()

    # Force bias to True (absorbing BN)
    config['use_bias'] = True
    config['name'] = f"{conv_layer.name}_fused"

    # Create new Conv layer with modified config
    if isinstance(conv_layer, layers.Conv2D):
        fused_layer = layers.Conv2D.from_config(config)
    elif isinstance(conv_layer, layers.Conv1D):
        fused_layer = layers.Conv1D.from_config(config)
    elif isinstance(conv_layer, layers.DepthwiseConv2D):
        fused_layer = layers.DepthwiseConv2D.from_config(config)
    else:
        raise ValueError(f"Unsupported Conv type: {type(conv_layer)}")

    # Apply to input
    output_tensor = fused_layer(input_tensor)

    # Set weights: copy original Conv weights + add dummy bias
    original_weights = conv_layer.get_weights()

    if len(original_weights) == 1:
        # Original Conv had no bias - add dummy bias
        kernel = original_weights[0]

        # Determine number of output channels based on layer type
        if isinstance(conv_layer, layers.DepthwiseConv2D):
            # DepthwiseConv2D: output channels = input_channels * depth_multiplier
            # Kernel shape for DepthwiseConv2D: (kernel_h, kernel_w, input_channels, depth_multiplier)
            num_filters = kernel.shape[2] * kernel.shape[3]
        else:
            # Regular Conv: output channels = last dimension of kernel
            num_filters = kernel.shape[-1]

        dummy_bias = np.zeros(num_filters, dtype=np.float32)
        fused_weights = [kernel, dummy_bias]
    else:
        # Original Conv had bias - keep it as-is
        fused_weights = original_weights

    fused_layer.set_weights(fused_weights)

    if verbose:
        print(f"  Created fused layer: {fused_layer.name}")
        print(f"    Original Conv bias: {conv_layer.use_bias}")
        print(f"    Fused Conv bias: True")
        print(f"    Weight shapes: {[w.shape for w in fused_weights]}")

    return fused_layer, output_tensor


def _clone_layer(layer):
    """Clone a layer with the same configuration."""
    config = layer.get_config()
    layer_class = type(layer)
    cloned = layer_class.from_config(config)

    # Copy weights if available
    if layer.get_weights():
        cloned.build(layer.input_shape)
        cloned.set_weights(layer.get_weights())

    return cloned
