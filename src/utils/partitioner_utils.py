'''
==================================================
File: partitioner_utils.py
Project: utils
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================

Special thanks to: 1. https://github.com/Lyken17/pytorch-OpCounter
                   2. https://github.com/MrYxJ/calculate-flops.pytorch
'''




"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = 
1. LAYER HOOKS: used to esimate the flops for a certain operation/layer.

Args:
    - input_shape: the shape of the input tensor to the layer
    - output_shape: the shape of the output tensor of the layer
    - layer : the instance of the layer for which we want to estimate the FLOPs
Return:
    - the number of FLOPs for the layer
    - the number of MACS for the layer
    
Note: we assume that hook output is a tuple of the form (FLOPs, MACs), where: MACs is the number of Multiply-Accumulate operations, and FLOPs is the number of floating point operations for the layer, but hooks are constructed in such way that MACs and FLOPs can be used interchangeably. MACs = 2 * FLOPs, since each MAC operation is a multiply followed by an add.
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = 
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
import prettytable as pt
import os
from typing import Dict, List
import pandas as pd


# Tranformation layers

def _calc_input(layer, input_shape, output_shape):
    return 0,0

def _calc_reshape(layer, input_shape, output_shape):
    return 0,0

def _calc_flatten(layer, input_shape, output_shape):
    return 0,0

def _calc_identity(layer, input_shape, output_shape):
    return 0,0

# Add layers
def _calc_add(layer, input_shape, output_shape):
    return np.prod(output_shape[1:]), 0

# Activation layers

def _calc_linear(layer, input_shape, output_shape):
    return 0,0

def _calc_relu(layer, input_shape, output_shape):
    return np.prod(output_shape[1:]), 0

def _calc_sigmoid(layer, input_shape, output_shape):
    return 4*np.prod(output_shape[1:]), 0

def _calc_tanh(layer, input_shape, output_shape):
    return 6*np.prod(output_shape[1:]), 0

def _calc_elu(layer, input_shape, output_shape):
    return 6*np.prod(output_shape[1:]), 0

def _calc_relu6(layer, input_shape, output_shape):
    return 2*np.prod(output_shape[1:]), 0

def _calc_leaky_relu(layer, input_shape, output_shape):
    return 4*np.prod(output_shape[1:]), 0

def _calc_softmax(layer, input_shape, output_shape):
    return (3*np.prod(output_shape[1:])) + 1, 0


def _calc_activation(layer, inputs_shape: tuple, outputs_shape: tuple):
    '''
    A hook serving as a router to the different activation functions.
    '''
    # Determine the type of activation function form the activation layer
    activation = layer.activation.__name__
    if activation == 'linear':
        return _calc_linear(layer, inputs_shape, outputs_shape)
    elif activation == 'relu':
        return _calc_relu(layer, inputs_shape, outputs_shape)
    elif activation == 'sigmoid':
        return _calc_sigmoid(layer, inputs_shape, outputs_shape)
    elif activation == 'tanh':
        return _calc_tanh(layer, inputs_shape, outputs_shape)
    elif activation == 'elu':
        return _calc_elu(layer, inputs_shape, outputs_shape)
    elif activation == 'relu6':
        return _calc_relu6(layer, inputs_shape, outputs_shape)
    elif activation == 'leakyrelu':
        return _calc_leaky_relu(layer, inputs_shape, outputs_shape)
    elif activation == 'softmax':
        return _calc_softmax(layer, inputs_shape, outputs_shape)

# Zero Padding layers

def _calc_zero_padding(layer, input_shape, output_shape):
    return 0,0

# Fully Connected layers

def _calc_fc(layer, input_shape, output_shape):
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0
    MACs  = np.prod(input_shape[1:]) * np.prod(output_shape[1:])
    FLOPs = np.prod(output_shape[1:]) * (2 * np.prod(input_shape[1:]) - 1) + bias_flops

    return FLOPs, MACs

# Pooling layers

def _calc_pool(layer, input_shape, output_shape):
    if isinstance(layer, layers.GlobalAveragePooling2D) or isinstance(layer, layers.GlobalAveragePooling1D):
        return np.prod(output_shape[1:]), 0
    k_size = np.prod(layer.pool_size)
    FLOPs = k_size - 1
    FLOPs *= np.prod(output_shape[1:])
    return FLOPs, 0


# Batch Normalization layers

def _calc_batch_norm(layer, input_shape, output_shape):
    norm_ops = 2 * output_shape[-1]
    norm_ops += 2 * np.prod(output_shape[1:])
    scale_ops = np.prod(output_shape[1:]) * 2 if layer.scale else 0
    bn_flops = norm_ops + scale_ops
    return bn_flops, 0

# Convolutional layers

def _calc_conv(layer, input_shape, output_shape):
    out_dims = output_shape[1:-1]
    out_channels = output_shape[-1]
    kernel_dims = layer.kernel_size
    in_channels = input_shape[-1]
    
    
    MACs = np.prod(kernel_dims) * np.prod(out_dims) * in_channels * out_channels
    FLOPs = 2 * np.prod(kernel_dims) * np.prod(out_dims) * in_channels * out_channels
    # Each output element gets a bias addition
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0

    MACs += bias_flops//2
    FLOPs += bias_flops

    return FLOPs, MACs

def _calc_depthwise_conv(layer, input_shape, output_shape):
    in_dims = input_shape[1:-1]
    in_channels = input_shape[-1]
    kernel_dims = layer.kernel_size
    out_dims = output_shape[1:-1]
    depth_multiplier = layer.depth_multiplier

    # Compute the number of operations required to covolve each channel with individual depthwise kernels, having 
    # depth_multiplier channels in the output
    MACs = np.prod(kernel_dims) * np.prod(out_dims) * in_channels
    FLOPs = 2 * np.prod(kernel_dims) * np.prod(out_dims) * in_channels
    # Concatenate the convoled outputs along the channel axis
    deph_flops = np.prod(out_dims) * (depth_multiplier - 1)
    # Add bias contributions
    bias_flops = np.prod(output_shape[1:]) if layer.use_bias else 0

    MACs += deph_flops//2 + bias_flops//2
    FLOPs += deph_flops + bias_flops

    return FLOPs, MACs

def _calc_transpose_conv(layer, input_shape, output_shape):
    # Numer of flops to determine the paddinng
    padding_flops = len(layer.kernel_size) * 8

    # Can also use layer.kernel_size, layer.filters and layer.input_shape to calculate FLOPs
    MACs, FLOPs = _calc_conv(layer, input_shape, output_shape)

    MACs += padding_flops//2
    FLOPs += padding_flops
    
    return FLOPs, MACs

# Dropout layers
def _calc_dropout(layer, input_shape, output_shape):
    return np.prod(input_shape[1:]), 0


register_hooks = {
    layers.InputLayer: _calc_input,
    layers.Reshape: _calc_reshape,
    layers.Flatten: _calc_flatten,
    layers.Add: _calc_add,
    layers.ZeroPadding1D: _calc_zero_padding,
    layers.ZeroPadding2D: _calc_zero_padding,
    layers.Dense: _calc_fc,
    layers.MaxPooling1D: _calc_pool,
    layers.MaxPooling2D: _calc_pool,
    layers.AveragePooling1D: _calc_pool,
    layers.AveragePooling2D: _calc_pool,
    layers.GlobalAveragePooling2D: _calc_pool,
    layers.GlobalAveragePooling1D: _calc_pool,
    layers.GlobalMaxPooling2D: _calc_pool,
    layers.GlobalMaxPooling1D: _calc_pool,
    layers.BatchNormalization: _calc_batch_norm,
    layers.Conv1D: _calc_conv,
    layers.Conv2D: _calc_conv,
    layers.DepthwiseConv2D: _calc_depthwise_conv,
    layers.DepthwiseConv1D: _calc_depthwise_conv,
    layers.Conv1DTranspose: _calc_transpose_conv,
    layers.Conv2DTranspose: _calc_transpose_conv,
    layers.Dropout: _calc_dropout,
    layers.Identity: _calc_identity,
    layers.ReLU: _calc_relu,
    layers.ELU: _calc_elu,
    layers.Activation: _calc_activation,
    'linear': _calc_linear,
    'relu': _calc_relu,
    'sigmoid': _calc_sigmoid,
    'tanh': _calc_tanh,
    'elu': _calc_elu,
    'relu6': _calc_relu6,
    'leakyrelu': _calc_leaky_relu,
    'softmax': _calc_softmax
}

"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
2. PARTITIONING STRATEGIES: used to partition the workload of a layer among the different PEs in the NoC.

We define 3 different ways of executing the partitioning of the model at layer-level-granularity:

- SPATIALLY: the partitioning is done by splitting the spatial dimensions of the layer among the different partitions:
            This means that each partition will own a subset of the spatial dimensions of the layer, together with the 
            weights and biases of all the layer's kernels. This approach is particulary useful when the input data is
            too large to fit in the memory of a single PE, while there are still a small number of input and output channels.

( The next two approaches are only valid for layers admitting a 4D input tensor, such as Conv2D, Conv2DTranspose, BatchNormalization, etc. ) 
       
- BY INPUT_CHANNELS: the partitioning is done by splitting the input channels of the layer among the different partitions:
                     This means that each partition will own a subset of the input channels of the layer, together with the 
                     corresponding weights and biases of all the layer's kernels for the chosen input channels. This approach is 
                     particularly useful when the input data is small enough to fit in the memory of a single PE, while there are
                     still a large number of input channels and small amount of output channels. It also induces the need for a 
                     communication phase between the partitions, in order to exchange output data

- BY OUTPUT_CHANNELS: the partitioning is done by splitting the output channels of the layer among the different partitions:
                     This means that each partition will own a subset of the output channels of the layer, together with the
                     corresponding weights and biases of the corresponding chosen kernels. This approach is particularly useful when
                     the input data is small enough to fit in the memory of a single PE, while there are still a large number of output
                     channels and small amount of input channels.

Since most of the time these approaches will be used in combination, we also define a fourth approach, which is a combination of the
previous three, using some tuning parameters:
    - in_sp: the percentage of the spatial dimensions of the layer that will be assigned a single partition
    - out_ch: the percentage of the output channels of the layer that will be assigned a single partition
    - in_ch: the percentage of the input channels of the layer that will be assigned a single partition

Using these parameters, we can define a custom partitioning strategy. In particular, choosing:
    - in_sp = x, out_ch = 0, in_ch = 0: will result in a pure spatial partitioning, where x% of the input is assgned to a single partition
    - in_sp = 0, out_ch = x, in_ch = 0: will result in a pure output channel partitioning, where x% of the output channels are assigned to a single partition
    - in_sp = 0, out_ch = 0, in_ch = x: will result in a pure input channel partitioning, where x% of the input channels are assigned to a single partition

* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
"""

from dataclasses import dataclass, field
from typing import List, Union, Tuple, Set, Dict
from copy import deepcopy
import string
import matplotlib.pyplot as plt
import graph as dg
import math

def get_pe_memory_size_from_config():
    """
    Read PE memory size from arch.json configuration file.
    Falls back to default if file not found or key missing.
    """
    try:
        import json
        import os
        # Try to read from the default arch.json file
        config_path = os.path.join(os.path.dirname(__file__), "../../config_files/arch.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('arch', {}).get('threshold_pe_mem', 256000)
    except Exception as e:
        print(f"Warning: Could not read PE memory size from config: {e}")
    
    # Fallback to default
    return 256000

@dataclass
class PE:
    """
    A simple dataclass to represent a Processing Element (PE) in the NoC. This will be just used to keep track
    of the PE's available resources (memory)
    """
    # The space used to store in memory a single number (input/weights/output) (in bytes)
    single_num: int = 2 #1 for int8, 2 for int16, 4 for int32/float32, 8 for float64
    # The size of the PE's memory (in bytes) - will be loaded from config
    mem_size: int = field(default_factory=get_pe_memory_size_from_config)
    # The amount of memory used by the PE (in bytes)
    mem_used: int = 0

    def __init__(self, mem_size: int = None):
        if mem_size is None:
            self.mem_size = get_pe_memory_size_from_config()
        else:
            self.mem_size = mem_size
        self.mem_used = 0

    def get_memory_occupation(self, as_percent: bool = False):
        """
        Returns the current memory occupation.
        Args:
            as_percent (bool): If True, returns occupation as a percentage of total memory.
        Returns:
            int or float: Memory used (bytes) or percentage.
        """
        if as_percent:
            return 100.0 * self.mem_used / self.mem_size if self.mem_size else 0.0
        return self.mem_used



# A function to construct the id to assign to each partition. The general rule for construction of the ID of 
# a partition is the following:
# for each type of partitioning, a unique number is assigned to the partition: since multiple partitions may be applied sequentially to the same layer, and since the order of
# in which the partitioning is applied is not important (the results produced are the same), we build the ID by concatenating the :
# - the layer number in the model,
# - the local ID of the spatial partitioning (if no partitioning is applied, the ID is x),
# - the local ID of the output channel partitioning (if no partitioning is applied, the ID is x),
# - the local ID of the input channel partitioning (if no partitioning is applied, the ID is x).
# As the order in which we perform the partitions is not important, we chose to always perform multiple partitions in the order: spatial-> ouput -> input.
# The "local" id of the partition is computed, based on the previous partition's local id


# A function to build the fully qualified ID of the partition
def _build_id(layer_name, id, partition_type, partition_local_id):
    id_list = id.split('-')
    # check that the layer name does not contain the "x" of "-" character
    if "-" in layer_name:
        raise ValueError("The layer name cannot contain the character '-'")

    if id_list[0] == 'x':
        id_list[0] = str(layer_name)
    else:
        assert id_list[0] == layer_name, "The layer name is different from the one in the ID"

    if partition_type == 'spatial':
        id_list[1] = str(partition_local_id)
    elif partition_type == 'output':
        id_list[2] = str(partition_local_id)
    elif partition_type == 'input':
        id_list[3] = str(partition_local_id)

    return '-'.join(id_list)   

# A datastructure to hold the information for the partition
@dataclass
class PartitionInfo:

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.id = 'x-x-x-x'
        self.task_id = None
        self.layer_id = None
        self.MACs = 0
        self.FLOPs = 0
        self.tot_size = 0
        for key, value in kwargs.items():
            setattr(self, key, value)
        

    def set_id(self, partition_type, partition_local_id, previous_id = None):
        if previous_id is not None:
            self.id = previous_id
        self.id = _build_id(self.layer.name, self.id, partition_type, partition_local_id)

    def set_split_factor(self, type, split_factor):
        if type == 'spatial':
            self.spatial_sp_fact = split_factor
        elif type == 'output':
            self.output_sp_fact = split_factor
        elif type == 'input':
            self.input_sp_fact = split_factor
        else:
            return ValueError("Invalid partition type")
    # Partition id
    id: str
    task_id: Union[None, int]

    # Layer name
    layer: keras.layers.Layer
    layer_id: Union[None, int]

    # Information on the partition
    # input bounds
    in_bounds: Union[None, List[tuple]] = None
    # output bounds
    out_bounds: Union[None, List[tuple]] = None
    # input channels included in the partiion
    in_ch: Union[None, List[tuple]] = None
    # output channels included in the partiton
    out_ch: Union[None, List[tuple]] = None
    # weights shape for the partition
    weights_shape: Union[None, List[tuple]] = None

    # additional data: used for merging nodes
    additional_data: Union[None, Dict] = None

    # mergeable flags
    mergeable: bool = False
    merger: bool = False

    # out-merging partition id
    out_merging: Union[None, str] = None

    # MACs
    MACs: int = 0
    # FLOPs
    FLOPs: int = 0
    # Total size of the partition
    tot_size: int = 0


def _split_spatial_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    '''
    A function to compute the spatial partitioning of the input and output tensors of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create -> the split factor is used to compute the number of partitions as a power of 2
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list
    '''

    def compute_receptive_field(region: list , layer):
            
            # region = [(x_min_out, y_min_out), (x_max_out, y_max_out)] or [(x_min_in,), (x_max_in,)]
            # layer: a Kernel layer object

            # If the layer instance is not convolutional, return the input region
            if  isinstance(layer, layers.InputLayer):
                return region
            elif not hasattr(layer, 'kernel_size'):
                # max pooling or average pooling or batch normalization
                if isinstance(layer, layers.MaxPooling2D) or isinstance(layer, layers.AveragePooling2D):
                    p_x = layer.pool_size[0]
                    p_y = layer.pool_size[1]
                    x_min = region[0][0] * p_x
                    x_max = region[1][0] * p_x
                    if len(region[0]) > 1:
                        y_min = region[0][1] * p_y
                        y_max = region[1][1] * p_y
                        return [(int(x_min), int(y_min)), (int(x_max), int(y_max))]
                    else:
                        return [(int(x_min),), (int(x_max),)]
                else:
                    return region 

            # Extract the kernel size, stride and padding
            # kernel_size = (k_x, k_y) or (k_x,)
            # stride = (s_x, s_y) or (s_x,)
            # padding = 'same' or 'valid'

            kernel_size = layer.kernel_size
            stride = layer.strides
            padding = layer.padding
            
            #convert the padding to a number
            padding = 0 if padding == 'valid' else 1

            # compute the receptive field
            x_min = - padding + region[0][0] * stride[0]
            x_max = - padding + region[1][0] * stride[0] + kernel_size[0] - 1
            if len(region[0]) > 1:
                y_min = - padding + region[0][1] * stride[1]
                y_max = - padding + region[1][1] * stride[1] + kernel_size[1] - 1
                return [(int(x_min), int(y_min)), (int(x_max), int(y_max))]
            return [(int(x_min),), (int(x_max),)]
    

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):

        new_partitions = []
        original_input_dims = deepcopy(input_dims)
        original_output_dims = deepcopy(output_dims)
        
        if len(input_dims) > 2:
            s_x = 2**((split_factor+1)//2)
            s_y = 2**(split_factor//2)
        else:
            s_x = 2**split_factor
            s_y = 1

        # Create an arrat of size s_x x s_y to store the output dimensions of the partitions: each object
        # is an array of size 2 (or 1 if the layer is Dense) containing the output dimensions of the partition
        # the array is initialized with zeros : [[0,0], [0,0], ...]
        output_dimensions = np.zeros((s_x, s_y), dtype=object)
        output_dimensions.fill([int(0), int(0)])
    

        # Append the partitions to the new list:
        granularity = 2**split_factor
        for _ in range(granularity):

            input_dims = deepcopy(original_input_dims)
            output_dims = deepcopy(original_output_dims)

            # compute the dimensions for the partitions
            if len(input_dims) > 2:
                # compute the dimensions for the partition on the output
                output_dims[0] = original_output_dims[0] if s_x == 1 else (original_output_dims[0]//s_x) + 1 if _%s_x < original_output_dims[0] % s_x else original_output_dims[0]//s_x
                output_dims[1] = original_output_dims[1] if s_y == 1 else (original_output_dims[1]//s_y) + 1 if _%s_y < original_output_dims[1] % s_y else original_output_dims[1]//s_y 

            else:
                # if the layer is Dense, we split the partition by considering only ouput neurons:
                index = 0
                output_dims[index] = original_output_dims[0] if s_x == 1 else (original_output_dims[0]//s_x) + 1 if _ < original_output_dims[0] % s_x else original_output_dims[0]//s_x

            out_x_0 = int(np.sum([output_dimensions[i, _//s_x][0] for i in range(0, _%s_x)])) if len(input_dims) > 2 else int(np.sum([output_dimensions[i][0] for i in range(0, _%s_x)]))
            out_bounds = [(max(out_x_0,0),), (min(out_x_0 + output_dims[0], original_output_dims[0]),)]
            # if isinstance(layer, layers.Dense) or (isinstance(layer, layers.Activation) and layer.activation.__name__ == 'softmax'):
            if len(input_dims) <= 2:
                in_bounds = [(0,), (original_input_dims[0],)]
                if isinstance(layer, layers.Dense):
                    weights_dims = [(original_input_dims[0], output_dims[0]),(output_dims[0],)] if layer.use_bias else [(original_input_dims[0], output_dims[0])]
            else:
                receptive_region = compute_receptive_field(out_bounds, layer)
                in_bounds = [(max(receptive_region[0][0],0),), (min(receptive_region[1][0], original_input_dims[0]),)]
            

            # check the validity of the partition
            if (out_bounds[1][0] - out_bounds[0][0] < 1) or (in_bounds[1][0] - in_bounds[0][0] < 1):
                print(f" The boundaries of the partition are invalid: {in_bounds} {out_bounds}")
                raise ValueError(f"Invalid partition for spatial dimensions, please decrease the split factor for layer: {layer.name}")
                
            if len(output_dims) > 2:
                out_y_0 = int(np.sum([output_dimensions[_%s_x, i][1] for i in range(0, _//s_x)])) 
                out_bounds = [(max(out_x_0, 0), max(out_y_0, 0)), (min(out_x_0 + output_dims[0], original_output_dims[0]), min(out_y_0 + output_dims[1], original_output_dims[1]))]
                #check the validity of the partition
                if out_bounds[1][1] - out_bounds[0][1] < 1:
                    print(f" The boundaries of the partition are invalid: {in_bounds} {out_bounds}")
                    raise ValueError(f"Invalid partition for spatial dimensions, please decrease the split factor for layer: {layer.name}")
                
            if len(input_dims) > 2 :
                receptive_region = compute_receptive_field(out_bounds, layer)
                in_bounds = [(max(receptive_region[0][0], 0), max(receptive_region[0][1], 0)), (min(receptive_region[1][0], original_input_dims[0]), min(receptive_region[1][1], original_input_dims[1]))]
                #check the validity of the partition
                if in_bounds[1][1] - in_bounds[0][1] < 1:
                    print(f" The boundaries of the partition are invalid: {in_bounds} {out_bounds}")
                    raise ValueError(f"Invalid partition for spatial dimensions, please decrease the split factor for layer: {layer.name}")

            output_dimensions[_%s_x, _//s_x] = output_dims

            cur = PartitionInfo(layer = layer, 
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = (0, input_dims[-1]) if len(input_dims) > 2 else None,
                                out_ch = (0, output_dims[-1]) if len(output_dims) > 2 else None,
                                weights_shape = weights_dims,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('spatial',  _ , partition.id if partition is not None else None)
            cur.set_split_factor('spatial', split_factor)
            new_partitions.append(cur)
        
        # print(f"Output dimensions: {output_dimensions}")
        return new_partitions

    

    # If the partitions array is empty, create a new one
    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape
        output_shape = layer.output.shape
        weights_dims = [w.shape for w in layer.get_weights()] #weight dimension do not change for convolutional layers
        
        
        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
        

    # If the partitions array is not empty, we apply the splitting on each of the partitions
    # new_partitions = []
    # for partition in partitions:
    #     input_bounds = partition.in_bounds
    #     output_bounds = partition.out_bounds
    #     input_dims = [input_bounds[1][i] - input_bounds[0][i] for i in range(len(input_bounds[0]))] if len(input_bounds[0]) > 1 else [input_bounds[1][0] - input_bounds[0][0]]
    #     input_dims = input_dims + [partition.in_ch] if partition.in_ch is not None else input_dims
    #     output_dims = [output_bounds[1][i] - output_bounds[0][i] for i in range(len(output_bounds[0]))] if len(output_bounds[0]) > 1 else [output_bounds[1][0] - output_bounds[0][0]]
    #     output_dims = output_dims + [partition.out_ch] if partition.out_ch is not None else output_dims
    #     weights_dims = [list(w) for w in partition.weights_shape] # weight dimensions do not change

    #     # Add the elements of the array return by _create_array to the new_partitions array
    #     new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    # return new_partitions


def _split_output_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    '''
    A function to compute the output channel partitioning of the output tensor of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list 
    '''

    # if the does not admit channels and the split factor is greater than 1, raise an error
    if len(layer.output.shape) < 3:
        return partitions

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):
        new_partitions = []
        
        weights_dims = deepcopy(weights_dims)

        if partition is None:
            output_dims = deepcopy(output_dims)
            num_output_ch = output_dims[-1]

            in_x_0 = 0
            in_bounds = [(in_x_0,), (input_dims[0],)]
            out_x_0 = 0
            out_bounds = [(out_x_0,), (output_dims[0],)]
            if len(input_dims) > 2:
                in_y_0 = 0
                in_bounds = [(in_x_0, in_y_0), (input_dims[0], input_dims[1])]

            if len(output_dims) > 2:
                out_y_0 = 0
                out_bounds = [(out_x_0, out_y_0), (output_dims[0], output_dims[1])]
            in_ch = (0, input_dims[-1]) if len(input_dims) > 2 else None
        else:
            num_output_ch = partition.out_ch[1] - partition.out_ch[0]
            output_dims = deepcopy(partition.out_bounds) + [num_output_ch]
            in_bounds = partition.in_bounds
            out_bounds = partition.out_bounds
            in_ch = partition.in_ch

        basic_step = num_output_ch // split_factor
        additions = num_output_ch % split_factor
        partition_index = 0

        # patition the output channels and number of kernels
        while output_dims[-1] > 0:

            ch_start = partition_index * basic_step + min(partition_index, additions)
            ch_end = (partition_index + 1) * basic_step + min(partition_index + 1, additions)
            #check partition validity
            if ch_end - ch_start < 1 or ch_end > num_output_ch:
                raise ValueError("Invalid partition for output channels, please decrease the split factor")
            weights_temp = []
            for i,weight in enumerate(weights_dims):
                index = -1
                weight[index] = ch_end - ch_start
                weights_temp.append(tuple(weight))

            cur = PartitionInfo(layer = layer,
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = in_ch,
                                out_ch = (ch_start, ch_end),
                                weights_shape = weights_temp,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('output', partition_index,  partition.id if partition is not None else None)
            cur.set_split_factor('output', split_factor)
            new_partitions.append(cur)

            output_dims[-1] -= ch_end - ch_start
            partition_index += 1


        return new_partitions


    # If the partitions array is empty, create a new one
    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape # input shape does not change
        output_shape = layer.output.shape
        weights_dims = [list(w.shape) for w in layer.get_weights()]

        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
    
    # If the partitions array is not empty, we apply the splitting on each of the partitions
    new_partitions = []
    for partition in partitions:
        input_dims = None
        output_dims = None

        weights_dims = [list(w) for w in partition.weights_shape] # weight dimensions do not change

        # Add the elements of the array return by _create_array to the new_partitions array
        new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    return new_partitions

    
def _split_input_dims(layer, split_factor, partitions: Union[None, List[PartitionInfo]] = None):
    """
    A function to compute the input channel partitioning of the input tensor of a layer.
    If a partitions list is given, for each object in the list, the partion strategy is applied to the object and a new list
    of partitions is created.

    Args:
    - layer : the layer to be partitioned
    - split_factor : the number of partitions to create
    - partitions : the partition list on which to apply the partitioning

    Returns:
    - a new PartitionInfo list
    """

    # if the does not admit channels and the split factor is greater than 1, raise an error
    if len(layer.input.shape) < 3:
        return partitions

    def _create_partitions(input_dims, output_dims, weights_dims, partition = None):
        new_partitions = []
        
        weights_dims = deepcopy(weights_dims)

        if partition is None:
            input_dims = deepcopy(input_dims)
            num_input_ch = input_dims[-1]
            in_x_0 = 0
            in_bounds = [(in_x_0,), (input_dims[0],)]
            out_x_0 = 0
            out_bounds = [(out_x_0,), (output_dims[0],)]
            if len(input_dims) > 2:
                in_y_0 = 0
                in_bounds = [(in_x_0, in_y_0), (input_dims[0], input_dims[1])]

            if len(output_dims) > 2:
                out_y_0 = 0
                out_bounds = [(out_x_0, out_y_0), (output_dims[0], output_dims[1])]
            out_ch = (0, output_dims[-1]) if len(output_dims) > 2 else None
        else:
            num_input_ch = partition.in_ch[1] - partition.in_ch[0]
            input_dims = deepcopy(partition.in_bounds) + [num_input_ch]
            in_bounds = partition.in_bounds
            out_bounds = partition.out_bounds
            out_ch = partition.out_ch

        basic_step = num_input_ch // split_factor
        additions = num_input_ch % split_factor
        partition_index = 0

        # partition the input channels and number of kernels
        while input_dims[-1] > 0:
            ch_start = partition_index * basic_step + min(partition_index, additions)
            ch_end = (partition_index + 1) * basic_step + min(partition_index + 1, additions)
            #check partition validity
            if ch_end - ch_start < 1 or ch_end > num_input_ch:
                raise ValueError("Invalid partition for input channels, please decrease the split factor")
            weights_temp = []
            for i,weight in enumerate(weights_dims):
                index = -2 if len(weight) > 2 else 0
                weight[index] = ch_end - ch_start
                weights_temp.append(tuple(weight))
            

            cur = PartitionInfo(layer = layer,
                                in_bounds = in_bounds,
                                out_bounds = out_bounds,
                                in_ch = (ch_start, ch_end),
                                out_ch = out_ch,
                                weights_shape = weights_temp,
                                spatial_sp_fact = partition.spatial_sp_fact if partition is not None else None, 
                                output_sp_fact = partition.output_sp_fact if partition is not None else None, 
                                input_sp_fact = partition.input_sp_fact if partition is not None else None)
            cur.set_id('input', partition_index, partition.id if partition is not None else None)
            cur.set_split_factor('input', split_factor)
            new_partitions.append(cur)

            input_dims[-1] -= ch_end - ch_start
            partition_index += 1

        return new_partitions



    if partitions is None:
        input_shape = layer.input[0].shape if type(layer.input) == list else layer.input.shape # input shape does not change
        output_shape = layer.output.shape # output shape does not change
        weights_dims = [list(w.shape) for w in layer.get_weights()]

        input_dims = list(input_shape[1:])
        output_dims = list(output_shape[1:])

        return _create_partitions(input_dims, output_dims, weights_dims)
    
    # If the partitions array is not empty, we apply the splitting on each of the partitions
    new_partitions = []
    for partition in partitions:
        input_dims = None
        output_dims = None
        weights_dims = [list(w) for w in partition.weights_shape]

        # Add the elements of the array return by _create_array to the new_partitions array
        new_partitions.extend(_create_partitions(input_dims, output_dims, weights_dims, partition))

    return new_partitions

def _build_partitions_from_layer(layer, spat = 0, out_ch = 1, in_ch = 1):
    """
    A function to split the workload of a layer among different partitions
    
    Args:
    - layer : the layer to be partitioned
    - spat : 'degree' of spatial partitioning
    - out_ch : number of partitions for the output channels
    - in_ch : number of partitions for the input channels

    Returns:
    - a list of PartitionInfo objects, each representing a partition of the layer

    """

    partitions = _split_spatial_dims(layer, spat)
    partitions = _split_output_dims(layer, out_ch, partitions)
    partitions = _split_input_dims(layer, in_ch, partitions)
    for p in partitions:
        p.FLOPs, p.MACs, p.tot_size = analyze_partition(p)
    return partitions

def _adaptive_parsel(layer, factor = 10, prev_computational_partitioning=None, chosen_splitting = "output", FLOP_threshold = 30000):
    """
    The functions selects the best partitioning strategy for a layer, based on the layer's characteristics.
    The main objective is to create the partitions for a layer with the lowest values for the splitting factors that, at the same time,
    also allows to host the partition on a single PE. This is done to aid convergence for the mapping algorithms that will be run on the task
    graph, as well as to reduce computation at such time. 
    It is important to notice that, by splitting the network to a finer granularity, the number of partitions will increase as well as the overall 
    number of communication (even if the whole amount of data being sent over the NoC should not increase considerably). This allows the solution space
    for the mapping system to grow in dimenions, eventually allowing for more optimal solutions to be found. At the same time, the complexity of the problem
    also increases.
    The strategy chosen to split the layer into different partitons is chosen based on the layer type, as well as on its input-output-weight shapes.
    The decision on whether to stop with the partitioning is taken based on the estimated current space needed to host the values necessary for the partition
    to be computed. If the space needed is greater than the available space on a single PE, the partitioning is stopped and the partition is not created.

    Enhanced function that selects the best partitioning strategy for a layer based on:
    1. For computational layers (Conv2D, Dense): FLOP-based balanced partitioning
    2. For auxiliary layers (BN, Activation, Pooling): Inherit from previous computational layer
    
    Args:
    - layer: the layer to be partitioned
    - prev_computational_partitioning: tuple (spatial, output, input) from the most recent computational layer
    - chosen_splitting: preferred splitting strategy for computational layers
    - FLOP_threshold: target FLOP count per partition for computational layers
    
    Notice: minimal splitting factor set is (0, 1, 1) for spatial, output, and input respectively.
    Max splitting factor to be achived is passed as factor parameter.
    
    Flop threshold here is also passed as a parameter! 
    
    Returns:
    - tuple (spatial, output, input): partitioning parameters for the layer
    """
    print("")
    print(f"====================================================")
    print(f"Adaptive partitioning for layer {layer.name} with FLOP threshold {FLOP_threshold}")
    
    # Configuration
    max_splitting_factor = factor # Maximum splitting factor for any dimension
    available_splitting = ['spatial', 'output', 'input']
    splitting_factors = {"spatial": 1, "output": 1, "input": 1}

    # Special case: InputLayer
    if isinstance(layer, layers.InputLayer):
        print("InputLayer detected - no partitioning needed")
        return 0, 1, 1
    
    # Classify layer type
    computational_layers = (layers.Conv2D, layers.Dense, layers.DepthwiseConv2D, layers.SeparableConv2D)
    auxiliary_layers = (
        layers.BatchNormalization, 
        layers.MaxPooling2D, 
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
        layers.GlobalAveragePooling2D,
        layers.Dropout,
        layers.Activation,
        layers.ReLU,
        layers.LeakyReLU
    )
    
    #in funtion build tasks graph we ommit Flatten and Add
    
    is_computational = isinstance(layer, computational_layers)
    is_auxiliary = isinstance(layer, auxiliary_layers)
    
    assert not (is_computational and is_auxiliary), "Layer cannot be both computational and auxiliary"
    #assert is_computational or is_auxiliary, "Layer must be either computational or auxiliary. Problem with the layer type classification"
    print(f"Layer classification - Computational: {is_computational}, Auxiliary: {is_auxiliary}")

    # Handle auxiliary layers - inherit from previous computational layer
    if is_auxiliary and prev_computational_partitioning is not None:
        
        spatial, output, input_split = prev_computational_partitioning
        print(f"Auxiliary layer inheriting partitioning from previous computational layer: ({spatial}, {output}, {input_split})")
        # Try to apply the inherited partitioning, with fallback mechanism
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Test if the current partitioning works
                test_partitions = _build_partitions_from_layer(layer, spatial, output, input_split)
                print(f"Successfully inherited partitioning: ({spatial}, {output}, {input_split})")
                return spatial, output, input_split
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: ValueError with partitioning ({spatial}, {output}, {input_split}): {e}")
                        
                # Step back strategy: reduce the largest splitting factor
                factors = [spatial, output, input_split]
                max_factor_idx = factors.index(max(factors))
                
                if max_factor_idx == 0 and spatial > 1:
                    spatial -= 1
                    print(f"Reduced spatial factor to {spatial}")
                elif max_factor_idx == 1 and output > 1:
                    output -= 1
                    print(f"Reduced output factor to {output}")
                elif max_factor_idx == 2 and input_split > 1:
                    input_split -= 1
                    print(f"Reduced input factor to {input_split}")
                else:
                    # If we can't reduce further, try reducing any factor > 1
                    if spatial > 1:
                        spatial -= 1
                        print(f"Fallback: reduced spatial factor to {spatial}")
                    elif output > 1:
                        output -= 1
                        print(f"Fallback: reduced output factor to {output}")
                    elif input_split > 1:
                        input_split -= 1
                        print(f"Fallback: reduced input factor to {input_split}")
                    else:
                        # All factors are 1, can't reduce further
                        print("Cannot reduce factors further, using (0, 1, 1)")
                        return 0, 1, 1
                
                attempt += 1
            
        # If we've exhausted all attempts, use minimal partitioning
        print(f"Warning: Could not find valid partitioning for auxiliary layer {layer.name} after {max_attempts} attempts")
        print("Using minimal partitioning (0, 1, 1)")
        return 0, 1, 1
        
    # Handle computational layers - FLOP-based partitioning
    if is_computational:
        print(f"Computational layer - applying FLOP-based partitioning")
        print(f"FLOP threshold: {FLOP_threshold}")
        
        # Determine available splitting strategies based on layer characteristics
        if isinstance(layer, layers.Dense):
            # Dense layers: primarily spatial splitting (batch dimension)
            available_splitting = ['spatial']
            chosen_splitting = 'spatial'
            print("Dense layer detected - using spatial splitting only")
        
        elif isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D, layers.SeparableConv2D)):
            # Conv layers: all splitting strategies available
            if len(layer.output.shape) < 4:  # If flattened conv output
                available_splitting = ['spatial']
                chosen_splitting = 'spatial'
            print(f"Convolutional layer detected - available splitting: {available_splitting}")
        
        # Remove unavailable splitting strategies
        if len(layer.output.shape) < 3:
            if 'output' in available_splitting:
                available_splitting.remove('output')
            if 'input' in available_splitting:
                available_splitting.remove('input')
        
        # Ensure chosen splitting is available
        if chosen_splitting not in available_splitting:
            chosen_splitting = available_splitting[0]
        
        print(f"Using splitting strategy: {chosen_splitting}")
        print(f"Available strategies: {available_splitting}")
        
        # Iteratively increase splitting factor until FLOP threshold is met
        iteration = 0
        max_iterations = 20  # Prevent infinite loops
                
        # Track which strategies have been tried and maxed out
        maxed_out_strategies = set()
        
        while iteration < max_iterations:
            try: 
                # Build partitions with current splitting factors
                partitions = _build_partitions_from_layer(
                    layer, 
                    splitting_factors['spatial'], 
                    splitting_factors['output'], 
                    splitting_factors['input']
                )
                
                # If successful, update last valid configuration
                last_valid_factors = splitting_factors.copy()
                
                # Check if all partitions meet FLOP threshold
                max_flops = max([p.FLOPs for p in partitions]) if partitions else 0
                avg_flops = sum([p.FLOPs for p in partitions]) / len(partitions) if partitions else 0
                
                #Edit here for 3 (or 4) different figures of merit
                
                print(f"Iteration {iteration}: Max FLOPs per partition: {max_flops}, Average: {avg_flops:.0f} with factors {splitting_factors}")
                
                if max_flops <= FLOP_threshold:
                    print(f"FLOP threshold satisfied!")
                    break
                
            except ValueError as e:
                print(f"ValueError encountered: {e}")
                print(f"Reverting {chosen_splitting} splitting factor from {splitting_factors[chosen_splitting]} to {last_valid_factors[chosen_splitting]}")
                
                # Revert to last valid configuration
                splitting_factors = last_valid_factors.copy()
                
                # Mark current strategy as maxed out and find next available one
                maxed_out_strategies.add(chosen_splitting)
                
                # Find next available strategy that hasn't been maxed out
                remaining_strategies = [s for s in available_splitting if s not in maxed_out_strategies]
                
                if remaining_strategies:
                    chosen_splitting = remaining_strategies[0]
                    print(f"Switching to splitting strategy: {chosen_splitting}")
                    iteration += 1
                    continue
                else:
                    print("All splitting strategies exhausted due to errors")
                    break
                
            # Increase splitting factor for chosen strategy
            if splitting_factors[chosen_splitting] >= max_splitting_factor:
                print(f"Maximum splitting factor reached for {chosen_splitting}")
                
                # Mark this strategy as maxed out
                maxed_out_strategies.add(chosen_splitting)
                
                # Find next available strategy that hasn't been maxed out
                remaining_strategies = [s for s in available_splitting if s not in maxed_out_strategies]
                
                # Try switching to next available strategy
                if remaining_strategies:
                    chosen_splitting = remaining_strategies[0]
                    print(f"Switching to splitting strategy: {chosen_splitting}")
                else:
                    print("All splitting strategies exhausted")
                    break
            else:
                splitting_factors[chosen_splitting] += 1
                print(f"Increased {chosen_splitting} splitting factor to {splitting_factors[chosen_splitting]}")
            
            iteration += 1
            
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations reached for layer {layer.name}")
    
    # Handle unknown layer types - use conservative defaults or inherit
    else:
        print(f"Unknown layer type : {layer.name} - using conservative approach")
        if prev_computational_partitioning is not None:
            spatial, output, input_split = prev_computational_partitioning
            print(f"Inheriting from previous computational layer: ({spatial}, {output}, {input_split})")
            return spatial, output, input_split
        else:
            print("No previous computational layer - using minimal partitioning")
            return 1, 1, 1
    
    result = (splitting_factors['spatial'], splitting_factors['output'], splitting_factors['input'])
    print(f"Final partitioning for {layer.name}: {result}")
    print("====================================================")
        
    return result


def _build_spatial_deps(partitions_layer1 : List[PartitionInfo], partitions_layer2: List[PartitionInfo], deps: Dict = None):
    """
    The function builds the dependencies between the partitions of two layers based on the spatial partitioning technique:
    in particular, it takes as input the partitions list, searches for the partitions of the two - already partitioned - layers 
    and builds the dependencies among this partition.
    OSS: it is assumed that the dependency between the two layers is already established
    and correct. This must be ensured at the moment of the creation of the dependency list among the layers

    Args:
    - partitions_layer1 : the partitions of the first layer
    - partitions_layer2 : the partitions of the second layer
    - deps : the dependencies between the partitions of the two layers

    Returns:
    - a dict of dependencies between the partitions of the two layers
    """
    partitions_1 = partitions_layer1
    partitions_2 = partitions_layer2

    # we then build the dependencies between the partitions: to do so, we must look at the spatial input and ouput dimensions of the partitions
    # of the two layers and check if they 'overlap'
    deps = {} if deps is None else deps
    for p1 in partitions_1:
        for p2 in partitions_2:
            
            # check if the partitions overlap:
            if len(p1.out_bounds[0]) > 1 and len(p2.in_bounds[0]) > 1:
                # Conv layers
                if p1.out_bounds[0][0] <= p2.in_bounds[1][0] and p1.out_bounds[1][0] >= p2.in_bounds[0][0] and p1.out_bounds[0][1] <= p2.in_bounds[1][1] and p1.out_bounds[1][1] >= p2.in_bounds[0][1]:
                        overlap = (min(p1.out_bounds[1][0], p2.in_bounds[1][0]) - max(p1.out_bounds[0][0], p2.in_bounds[0][0])) * (min(p1.out_bounds[1][1], p2.in_bounds[1][1]) - max(p1.out_bounds[0][1], p2.in_bounds[0][1]))
                        if overlap >0:
                            if deps.get((p1.id, p2.id)) is not None:
                                deps[(p1.id, p2.id)] += overlap
                            else:
                                deps[(p1.id, p2.id)] = overlap
            elif len(p1.out_bounds[0]) == 1 and len(p2.in_bounds[0]) == 1:
                # 1D layers
                # OSS: for the dense layers, depependencies are always present between the partitions of a layer and the partitions of the next layer,
                # in partitcular, the communication size is equal to the number of output neurons of the partition of the first layer
                if p1.out_bounds[0][0] <= p2.in_bounds[1][0] and p1.out_bounds[1][0] >= p2.in_bounds[0][0]:
                    overlap = (min(p1.out_bounds[1][0], p2.in_bounds[1][0]) - max(p1.out_bounds[0][0], p2.in_bounds[0][0]))
                    if overlap >0:
                        if deps.get((p1.id, p2.id)) is not None:
                            deps[(p1.id, p2.id)] += overlap
                        else:
                            deps[(p1.id, p2.id)] = overlap
            elif len(p1.out_bounds[0]) > 1 and len(p2.in_bounds[0]) == 1:
                # Border case: 2D layer -> 1D layer
                # in this case, we assume that all the outputs of the 2D layer are connected to the 1D layer
                
                overlap = (p1.out_bounds[1][0] - p1.out_bounds[0][0]) * (p1.out_bounds[1][1] - p1.out_bounds[0][1])
                overlap *= p1.out_ch[1] - p1.out_ch[0]
                if overlap >0:
                    input_size = p2.in_bounds[1][0] - p2.in_bounds[0][0]
                    if deps.get((p1.id, p2.id)) is not None:
                        deps[(p1.id, p2.id)] += overlap
                    else:
                        deps[(p1.id, p2.id)] = overlap
            else :
                raise Exception("Spatial dependencies for 1D -> 2D layers not implemented")

    return deps

def _build_outin_deps(partitions_layer1: List[PartitionInfo], partitions_layer2: List[PartitionInfo], deps: Dict = None):
    """
    A function to build the dependencies between the partitions of two layers based on the input/output channel partitioning technique.

    Args:
    - partitions_layer1 : the partitions of the first layer
    - partitions_layer2 : the partitions of the second layer
    - deps : a dictionary of dependencies between the partitions of the layers

    Returns:
    - a dict of dependencies between the partitions of the two layers
    """

    partitions_1 = partitions_layer1
    partitions_2 = partitions_layer2

    # we then build the dependencies between the partitions of the two layers
    deps = {} if deps is None else deps
    for p1 in partitions_1:
        for p2 in partitions_2:

            if p1.out_ch is not None and p2.in_ch is not None:
                # check for the overlap between the output channels of the first layer and the input channels of the second layer.
                # OSS: no need to check that the input channels assigned to the partition since it is assumed that, indipendently from 
                # the input channels assigned, the partial results will still need to be reduced and passed to the next layer, thus
                # creating a dependency between the partitions. Furthermore, also the number of input channels assigned to the partition
                # is not relevant in terms of communication weight, since berfore performing the computation, we can simply reduce the partial
                # results corresponding to different input channels, thus sending a number of bytes corresponding to a single channel (2D) tensor.
                if p1.out_ch[0] <= p2.in_ch[1] and p1.out_ch[1] >= p2.in_ch[0]:
                    overlap = min(p1.out_ch[1], p2.in_ch[1]) - max(p1.out_ch[0], p2.in_ch[0])
                    if overlap >0:
                        if deps.get((p1.id, p2.id)) is not None:
                            deps[(p1.id, p2.id)] *= overlap
                        # else:
                        #     deps[(p1.id, p2.id)] = overlap
                

    return deps


def _build_layer_deps(model: keras.Model)->Set:
    """"
    A function to infer the dependencies between the layers of a keras model: it looks at the model's dependency graph and builds
    a list of dependencies between the layers of the model. Those are then used to build the dependencies between the partitions of the layers.

    Args:
    - model : the keras model for which to infer the dependencies

    Returns:
    - a set of dependencies between the layers of the model
    """

    dependencies = set()
    dep_id = 0

    def add_to_deps(node_in, layer_out, deps, dep_id):
        if isinstance(node_in.inbound_layers, list):
            for in_layer in node_in.inbound_layers:
                # Skip Flatten and Add layers (Add is handled separately)
                if isinstance(in_layer, (layers.Flatten, layers.Add)):
                    continue

                deps.add((dep_id, in_layer, layer_out))
        else:
            in_layer = node_in.inbound_layers
            # Skip Flatten and Add layers (Add is handled separately)
            if isinstance(in_layer, (layers.Flatten, layers.Add)):
                return
            deps.add(( dep_id, in_layer, layer_out))


    for layer in model.layers:
        # Special handling for Flatten: create dependencies between its input and output nodes
        if isinstance(layer, layers.Flatten):
            for node_in in layer._inbound_nodes:
                for node_out in layer._outbound_nodes:
                    out_layer = node_out.outbound_layer # just one output layer
                    add_to_deps(node_in, out_layer, dependencies, dep_id)
                    dep_id += 1

        # Special handling for Add: create dependencies from ALL input branches to next layer
        # This implements residual/skip connections where multiple branches merge
        elif isinstance(layer, layers.Add):
            # Get all input branches to the ADD layer
            input_layers = []
            for node_in in layer._inbound_nodes:
                if isinstance(node_in.inbound_layers, list):
                    input_layers.extend(node_in.inbound_layers)
                else:
                    input_layers.append(node_in.inbound_layers)

            # Get all output layers that ADD connects to
            output_layers = []
            for node_out in layer._outbound_nodes:
                output_layers.append(node_out.outbound_layer)

            # Create dependencies: each input branch → each output layer
            # This creates a fan-in pattern where all branches feed into the next layer
            for in_layer in input_layers:
                for out_layer in output_layers:
                    dependencies.add((dep_id, in_layer, out_layer))
                    dep_id += 1

        # Regular layers: standard dependency handling
        else:
            for node in layer._inbound_nodes:
                add_to_deps(node, layer, dependencies, dep_id)
                dep_id += 1


    return dependencies


def _build_partitions_deps(partitions_layer1 : List[PartitionInfo], partitions_layer2 : List[PartitionInfo], layer_to_layer_set : Set, deps: Dict = None)-> Dict:
    """
    Given the dictionary of partitions and the list of dependent layers, the function builds the dependencies between the partitions of the layers
    included in the layer_to_layer_list.
    
    Args:
    - partitions : a dictionary of partitions of the layers
    - layer_to_layer_list : a list of of tuples representing the dependent layers

    Returns:
    - a dictionary of dependencies between the partitions of the layers
    """

    layer_name1 = partitions_layer1[0].id.split('-')[0]
    layer_name2 = partitions_layer2[0].id.split('-')[0]

    if deps is None:
        deps = {}

    # check if the layers are dependent, we do not care about the layer_dep_id
    for _, layer1, layer2 in layer_to_layer_set:

        if layer1.name == layer_name1 and layer2.name == layer_name2:
            # build the dependencies between the partitions of the two layers
            deps[(layer_name1, layer_name2)] = _build_spatial_deps(partitions_layer1, partitions_layer2)
            deps[(layer_name1, layer_name2)] = _build_outin_deps(partitions_layer1, partitions_layer2, deps[(layer_name1, layer_name2)])

    return deps

def _group_partitions(partitions_layer1 : List[PartitionInfo], partitions_layer2 : List[PartitionInfo], layer_to_layer_set: Set, deps: Dict)-> None:
    """
    The function groups together partitions that are interdependent only with each other:
    this reduces the number of partitions, aiding the mapping algorithms to convergence.
    The two partitions are substituted by a single partition whose total size and MACs/FLOPs
    are the sum of the two partitions.
    
    Args:
    - partition1 : the first partition to group
    - partition2 : the second partition to group

    Returns:
    - a PartitionInfo object representing the grouped partitions

    """
    name_layer1 = partitions_layer1[0].id.split('-')[0]
    partitions1_map_to_int = {p.id:i for i, p in enumerate(partitions_layer1)}
    partitions2_map_to_int = {p.id:i for i, p in enumerate(partitions_layer2)}
    name_layer2 = partitions_layer2[0].id.split('-')[0]
        

    def _mark_partitions(part1: PartitionInfo, part2: PartitionInfo):
        """
        Handy function used to mark a partition as mergeable with another one
        """
        # if the first partition is already marked as mergeable, we mark the second partition as mergeable too,
        # and set the out_merging field of the first partition to the second partition
        if part1.mergeable is True:
            assert part2.out_merging is None and part2.mergeable is False and part1.merger is False and part1.mergeable is True
            part1.out_merging = part2.id
            part2.mergeable = True
        else:
            assert part1.mergeable is False and part2.mergeable is False and part2.merger is False and part1.merger is False
            part1.out_merging = part2.id
            part1.merger = True
            part2.mergeable = True

    # build the dependencies between the partitions of the two layers
    temp_deps = _build_partitions_deps(partitions_layer1, partitions_layer2, layer_to_layer_set)

    # Check if there are actual dependencies between these layers
    if (name_layer1, name_layer2) not in temp_deps:
        # No dependencies between these layers (they are parallel), skip grouping
        print(f"No dependencies found between {name_layer1} and {name_layer2} - skipping grouping (parallel layers)")
        return

    # create the connectivity matrix for clearer dependencies visualization
    connectivity_matrix = np.zeros((len(partitions_layer1), len(partitions_layer2)))

    for p1,p2 in temp_deps[(name_layer1, name_layer2)]:
        connectivity_matrix[partitions1_map_to_int[p1], partitions2_map_to_int[p2]] += 1 if temp_deps[(name_layer1, name_layer2)][p1,p2] > 0 else 0

    # for each couple of partitions, check if they are interdependent
    # only between themselves
    for i, p1 in enumerate(partitions_layer1):
        for j, p2 in enumerate(partitions_layer2):
            # check that the sum over the row and the column are both equal to 1
            if connectivity_matrix[i,:].sum() == 1 and connectivity_matrix[:,j].sum() == 1 and connectivity_matrix[i,j] == 1 and ("input" not in p1.id):
                # mark the partitions
                _ = _mark_partitions(p1, p2)

    # build the dependencies between the partitions of the two layers
    deps[(name_layer1, name_layer2)] = temp_deps[(name_layer1, name_layer2)]

def _section_partitions(partitions: Dict[str, List[PartitionInfo]], partitions_deps: Dict[Tuple[str, str], int], memory_constraints: int):
    """
    This function is used to section the groups of partitions if the total size of the group does not satisfy the memory constraints of the PE:
    a very simle implementation would be to go over the partitions in the group and stack as much as we possibly
    can in the same group, then move to the next group. If we assume the "trend" of communication to be
    monotonic decreasing, this should be a good approximation for an optimal solution.
    On the other hand, if such approximation is not valid, we have basically a min cut problem with additional constraints:
    to solve this, we could use a greedy algorithm with dynamic programming.
    """

    def compute_group_size(partitions_group: List[PartitionInfo]):
        """
        A function to compute the total size of a group of partitions
        """
        new_tot_size = 0
        input_size = np.prod([partitions_group[0].in_bounds[1][i] - partitions_group[0].in_bounds[0][i] for i in range(len(partitions_group[0].in_bounds[0]))] if len(partitions_group[0].in_bounds[0]) > 1 else [partitions_group[0].in_bounds[1][0] - partitions_group[0].in_bounds[0][0]])
        output_size = 0
        additional_data = {}
        for p in partitions_group:
            additional_data[p.id] = p
            for w in p.weights_shape:
                new_tot_size += np.prod(w)
            output_size += np.prod([p.out_bounds[1][i] - p.out_bounds[0][i] for i in range(len(p.out_bounds[0]))] if len(p.out_bounds[0]) > 1 else [p.out_bounds[1][0] - p.out_bounds[0][0]])
        new_tot_size += input_size + output_size
        return new_tot_size

    # go over the partitions: for each partitions, check if that partition is the head (merger) of a group of partitions
    # if so, go down the chain of partitions and save them in a list
    
    for layer_name, partitions_list in partitions.items():
        for p in partitions_list:
            if p.merger is True:
                to_group = []
                to_group.append(p)
                next_p = p
                while next_p.out_merging is not None:
                    layer_name = next_p.out_merging.split('-')[0]
                    for p in partitions[layer_name]:
                        if p.id == next_p.out_merging:
                            next_p = p
                            break
                    to_group.append(next_p)
                
                # check the total size of the group of partitions: if it is smaller than the memory constraints, we can keep the group as it is
                # otherwise, we split the group

                if compute_group_size(to_group) <= memory_constraints:
                    return 
                else:
                    new_groups = []
                    cur_group = []
                    cur_group_size = 0
                    for p in to_group:
                        if compute_group_size([p]) > memory_constraints:
                            raise ValueError("Partition size is greater than the memory constraints of the PE")
                        cur_group_size += compute_group_size([p])
                        if cur_group_size <= memory_constraints:
                            cur_group.append(p)
                        else:
                            new_groups.append(cur_group)
                            cur_group = [p]
                            cur_group_size = compute_group_size([p])
                    new_groups.append(cur_group)

                    # now for each sub-group, we access the partitions and change the mergeable, merger and out_merging fields
                    # as follows:
                    # - if the partition is the first in the group, we set the mergeable field to False and the merger field to True
                    # - if the partition is the last in the group, we set the mergeable field to True and the out_merging field to None

                    for group in new_groups:
                        for p_id, p in enumerate(group):
                            if p_id == 0:
                                p.mergeable = False
                                p.merger = True
                                p.out_merging = p.out_merging if len(group) > 1 else None
                            elif p_id == len(group) - 1:
                                p.mergeable = True
                                p.out_merging = None
                            else:
                                p.mergeable = True
                                p.merger = False
                                # p.out_merging = group[p_id+1].id

    return 



def _build_straight_through_deps(partitions: Dict[str, List[PartitionInfo]], partitions_deps: Dict[Tuple[str, str], int])-> Tuple[Dict, Dict]:
    """
    The function goes over the partitions, looks for any grouping markings and effectively groups the partitions by creating
    a unique partition
    """


    def _merge_partitions(partition_to_merge: List[PartitionInfo], partitions_deps: Dict):
        """
        Handy function to group together partitions
        """
    
        new_id = partition_to_merge[0].id
        new_layer = partition_to_merge[0].layer
        new_in_bounds = partition_to_merge[0].in_bounds
        new_out_bounds = partition_to_merge[-1].out_bounds
        new_in_ch = partition_to_merge[0].in_ch
        new_out_ch = partition_to_merge[-1].out_ch
        new_weights_shape = [w for p in partition_to_merge for w in p.weights_shape]
        # compute the total size of the partition
        new_MACs = sum([p.MACs for p in partition_to_merge])
        new_FLOPs = sum([p.FLOPs for p in partition_to_merge])
        # the total size is computed as the sum of the sizes of the weight for the partitions
        # and the sum of the maximum size of the input and output tensors
        new_tot_size = 0
        input_size = np.prod([partition_to_merge[0].in_bounds[1][i] - partition_to_merge[0].in_bounds[0][i] for i in range(len(partition_to_merge[0].in_bounds[0]))] if len(partition_to_merge[0].in_bounds[0]) > 1 else [partition_to_merge[0].in_bounds[1][0] - partition_to_merge[0].in_bounds[0][0]])
        output_size = 0
        additional_data = {}
        for p in partition_to_merge:
            additional_data[p.id] = p
            for w in p.weights_shape:
                new_tot_size += np.prod(w)
            output_size += np.prod([p.out_bounds[1][i] - p.out_bounds[0][i] for i in range(len(p.out_bounds[0]))] if len(p.out_bounds[0]) > 1 else [p.out_bounds[1][0] - p.out_bounds[0][0]])
        new_tot_size += input_size + output_size
        
        new_partition = PartitionInfo(layer = new_layer,
                                    id = new_id,
                                    in_bounds = new_in_bounds,
                                    out_bounds = new_out_bounds,
                                    in_ch = new_in_ch,
                                    out_ch = new_out_ch,
                                    weights_shape = new_weights_shape,
                                    FLOPs = new_FLOPs,
                                    MACs = new_MACs,
                                    tot_size = new_tot_size,
                                    additional_data = additional_data)
        
        # delete the dependencies between the partitions that are going to be merged
        for p_id, p in enumerate(partition_to_merge):
            if p_id == 0:
                continue
            prev_p = partition_to_merge[p_id-1]
            p_layer_name = p.id.split('-')[0]
            prev_p_layer_name = prev_p.id.split('-')[0]
            # delete the related dependecies
            del partitions_deps[(prev_p_layer_name, p_layer_name)][(prev_p.id, p.id)]
            if len(partitions_deps[(prev_p_layer_name, p_layer_name)]) == 0:
                del partitions_deps[(prev_p_layer_name, p_layer_name)]

        # find the element in the partitions_deps that has as first element in the key the id of the last partition in the partition_to_merge list
        stitching_deps = {}
        new_key = None
        for key in partitions_deps.keys():
            if key[0] == partition_to_merge[-1].id.split('-')[0]:
                stitching_deps = deepcopy(partitions_deps[key])
                pre_key = key
                new_key = (new_id.split('-')[0], key[1])
                break

        # stitch back together the dependencies
        if new_key is not None:
            partitions_deps[new_key] = {} if partitions_deps.get(new_key) is None else partitions_deps[new_key]
            for key, value in stitching_deps.items():
                if key[0] == partition_to_merge[-1].id:
                    del partitions_deps[pre_key][key]
                    partitions_deps[new_key][(new_id, key[1])] = value
            if len(partitions_deps[pre_key]) == 0:
                del partitions_deps[pre_key]

        return new_partition

    # go over the partitions and, if one of them is marked as mergeable, go down the chain of partitions to create a single partition
    
    for layer_name, partitions_list in partitions.items():
        new_partitions = []
        for p in partitions_list:
            to_group = []
            if p.mergeable is False and p.merger is False:
                new_partitions.append(p)
            elif p.mergeable is True:
                assert p.merger is False
                # delete the partitions that are marked as mergeable
                # (SIMPLY DON'T APPEND TO THE NEW PARTITIONS)
                continue
            elif p.merger is True:
                # until no more mergiable partitions are found,
                # go down the chain of partitions
                # assert p.out_merging is not None
                to_group.append(p)
                next_p = p
                while next_p.out_merging is not None:
                    next_layer_name = next_p.out_merging.split('-')[0]
                    next_partitions = partitions[next_layer_name]
                    # find the partition in the list that has the same id as the out_merging field of the partition
                    next_p = None
                    for part in next_partitions:
                        if part.id == to_group[-1].out_merging:
                            to_group.append(part)
                            next_p = part
                            break   
                # create the new partition from the partitions in to_group
                new_partition = _merge_partitions(to_group, partitions_deps)
                new_partitions.append(new_partition)
        partitions[layer_name] = new_partitions
    
    return partitions, partitions_deps

def build_partitions(model: keras.Model, grid, chosen_splitting_strategy: str = "spatial", grouping: bool = True, verbose : bool = True)-> Tuple[dict, dict]:
    """
    The function creates the partitions for each layer in the model, based on the partitioning strategies defined above.

    Args:
    - model : the model to partition

    Returns:
    - a dict of partitions: the key is the layer name and the svalue is a list of PartitionInfo objects, each representing a partition of the layer
    - a dict of dependencies between the partitions of the layers
    """
    
    max_splitting_factor = 5

    partitions = {}
    partitions_deps = {}
    pe = PE()
    nPEs = grid.K * grid.K # number of PEs in the grid
    print(f"PE memory size: {pe.mem_size} and number of PEs: {nPEs}", )
    layer_deps = _build_layer_deps(model)
    
    #avilable splitting strategies spatial, output, input:: 
    assert chosen_splitting_strategy in ('spatial', 'output', 'input'), \
        f"Invalid chosen_splitting_strategy: {chosen_splitting_strategy}. Must be 'spatial', 'output', or 'input'."
        
    chosen_splitting = chosen_splitting_strategy
    
    # Track the most recent computational layer's partitioning
    last_computational_partitioning = None
    computational_layers = (layers.Conv1D, layers.Conv2D, layers.Conv3D, layers.Dense, layers.DepthwiseConv2D, layers.DepthwiseConv1D, layers.SeparableConv2D)
    
    layers_skip = (layers.Flatten, layers.Reshape, layers.Add, layers.Identity)
    
    # first run for the input layer
    input_layer = model.layers[0]
    
    # for input layer use defalut splitting stategy and Flops threshold
    spat, out_ch, in_ch = _adaptive_parsel(input_layer, factor = max_splitting_factor, chosen_splitting = chosen_splitting, prev_computational_partitioning = last_computational_partitioning)
    
    partitions[input_layer.name] = _build_partitions_from_layer(input_layer, spat, out_ch, in_ch)
    if verbose:
        print("Layer {} partitioned succesfully with partition parameters: {} ".format(input_layer.name, (spat, out_ch, in_ch)))

    actual_prev_layer = None  # Track the actual previous non-skipped layer
    # then proceed with the other layers
    n = 2 # multiplying PEs factor for FLOPs threshold
    for _, prev_layer, layer in sorted(layer_deps, key = lambda x: x[0]):
        
        # if the layer is in a skipped layers list we can direclty skip it
        if isinstance(layer, layers_skip):
            print("Skipping layer {}".format(layer.name))
            print("Warning: Layer {} is skipped, might be a problem with dependencies.".format(layer.name))
            continue
        
        FLOPs, MACs = _analyze_layer(layer)
        
        # 1. strategy: Equal FLOPs per partitions across layer 
        # set FLOPS threshold per layer according to number of PEs and MACs per layer!
        Flops_theshold = FLOPs / (nPEs * n) if isinstance(layer, computational_layers) else 1
        
        #2. strategy: equal FLOPs per each partition across network
        
        #3. Data IN same per each partition
        
        #4. Data OUT same per each partition
        
        #5. Max utilization of the PEs Memory (equally distributed across partitions) (later to check with the mapping)
        
        if verbose:
            print("")
            print(f"Analyzing layer {layer.name} with FLOPs: {FLOPs}, MACs: {MACs}, FLOP threshold: {Flops_theshold}")
    
        spat, out_ch, in_ch = _adaptive_parsel(layer, factor = max_splitting_factor, prev_computational_partitioning = last_computational_partitioning, chosen_splitting = chosen_splitting, FLOP_threshold = Flops_theshold)
        partitions[layer.name] = _build_partitions_from_layer(layer, spat, out_ch, in_ch)
        
        # Update tracking of computational layers
        if isinstance(layer, computational_layers):
            last_computational_partitioning = (spat, out_ch, in_ch)
            if verbose:
                print(f"Updated computational layer tracking: {last_computational_partitioning}")
        
        # group the partitions that are interdependent
        # Use actual_prev_layer if available, otherwise fall back to prev_layer
        if actual_prev_layer is not None:
            partitions1 = partitions[actual_prev_layer.name]
        else:
            partitions1 = partitions[prev_layer.name]
        
        partitions2 = partitions[layer.name]
        # dependencies are delt directly in _group_partitions
        _group_partitions(partitions1, partitions2, layer_deps, partitions_deps)
        
        # Update the actual previous layer to current layer (since it wasn't skipped)
        actual_prev_layer = layer
        
        if verbose:
            print("Layer {} partitioned succesfully with partition parameters: {} ".format(layer.name, (spat, out_ch, in_ch)))

    if grouping:
        _section_partitions(partitions, partitions_deps, pe.mem_size)
        partitions, partitions_deps = _build_straight_through_deps(partitions=partitions, partitions_deps=partitions_deps)

    return partitions, partitions_deps


def build_partitions_splitting_input(model: keras.Model, grid, partitioning_tuple = (int, int, int), grouping: bool = True, verbose : bool = True)-> Tuple[dict, dict]:
    """
    The function creates the partitions for each layer in the model, based on the partitioning strategies provided as input.

    Args:
    - model : the model to partition

    Returns:
    - a dict of partitions: the key is the layer name and the svalue is a list of PartitionInfo objects, each representing a partition of the layer
    - a dict of dependencies between the partitions of the layers
    """
    
    partitions = {}
    partitions_deps = {}
    pe = PE()
    nPEs = grid.K * grid.K # number of PEs in the grid
    print(f"PE memory size: {pe.mem_size} and number of PEs: {nPEs}", )
    layer_deps = _build_layer_deps(model)
    
    spat, out_ch, in_ch = partitioning_tuple
    
    # Track the most recent computational layer's partitioning
    last_computational_partitioning = None
    computational_layers = (layers.Conv1D, layers.Conv2D, layers.Conv3D, layers.Dense, layers.DepthwiseConv2D, layers.DepthwiseConv1D, layers.SeparableConv2D)
    
    layers_skip = (layers.Flatten, layers.Reshape, layers.Add, layers.Identity)
    
    # first run for the input layer
    input_layer = model.layers[0]
    
    partitions[input_layer.name] = _build_partitions_from_layer(input_layer, 0, 1, 1)
    if verbose:
        print("Layer {} partitioned succesfully with partition parameters: {} ".format(input_layer.name, (0, 1, 1)))

    actual_prev_layer = None  # Track the actual previous non-skipped layer
    # then proceed with the other layers
    for _, prev_layer, layer in sorted(layer_deps, key = lambda x: x[0]):
        
        # if the layer is in a skipped layers list we can direclty skip it
        if isinstance(layer, layers_skip):
            print("Skipping layer {}".format(layer.name))
            print("Warning: Layer {} is skipped, might be a problem with dependencies.".format(layer.name))
            continue
        
        partitions[layer.name] = _build_partitions_from_layer(layer, spat, out_ch, in_ch)
        
        # Update tracking of computational layers
        if isinstance(layer, computational_layers):
            last_computational_partitioning = (spat, out_ch, in_ch)
            if verbose:
                print(f"Updated computational layer tracking: {last_computational_partitioning}")
        
        # group the partitions that are interdependent
        # Use actual_prev_layer if available, otherwise fall back to prev_layer
        if actual_prev_layer is not None:
            partitions1 = partitions[actual_prev_layer.name]
        else:
            partitions1 = partitions[prev_layer.name]
        
        partitions2 = partitions[layer.name]
        # dependencies are delt directly in _group_partitions
        _group_partitions(partitions1, partitions2, layer_deps, partitions_deps)
        
        # Update the actual previous layer to current layer (since it wasn't skipped)
        actual_prev_layer = layer
        
        if verbose:
            print("Layer {} partitioned succesfully with partition parameters: {} ".format(layer.name, (spat, out_ch, in_ch)))

    if grouping:
        _section_partitions(partitions, partitions_deps, pe.mem_size)
        partitions, partitions_deps = _build_straight_through_deps(partitions=partitions, partitions_deps=partitions_deps)

    return partitions, partitions_deps

def build_partitions_splitting_input_for_many_tuples(model: keras.Model, grid, partitioning_tuple=None, grouping: bool = True, verbose: bool = True) -> Tuple[dict, dict]:
    """
    The function creates the partitions for each layer in the model, based on the partitioning strategies provided as input.

    Args:
    - model: the model to partition
    - partitioning_tuple: either a single tuple (spat, out_ch, in_ch) for all layers,
                         or a list of tuples with one tuple per layer

    Returns:
    - a dict of partitions: the key is the layer name and the value is a list of PartitionInfo objects
    - a dict of dependencies between the partitions of the layers
    """
    
    partitions = {}
    partitions_deps = {}
    pe = PE()
    nPEs = grid.K * grid.K  # number of PEs in the grid
    print(f"PE memory size: {pe.mem_size} and number of PEs: {nPEs}")
    layer_deps = _build_layer_deps(model)
    
    # Handle different input types for partitioning_tuple
    if partitioning_tuple is None:
        # Default partitioning if none provided
        partitioning_tuple = [(1, 1, 1)] * len(model.layers)
    elif isinstance(partitioning_tuple, tuple) and len(partitioning_tuple) == 3:
        # Single tuple provided - use for all layers
        partitioning_tuple = [partitioning_tuple] * len(model.layers)
    elif isinstance(partitioning_tuple, list):
        # List provided - check if it matches the number of layers
        if len(partitioning_tuple) != len(model.layers):
            print(f"Warning: partitioning list length ({len(partitioning_tuple)}) doesn't match number of layers ({len(model.layers)})")
            # Extend or truncate as needed
            if len(partitioning_tuple) < len(model.layers):
                partitioning_tuple = partitioning_tuple + [(1, 1, 1)] * (len(model.layers) - len(partitioning_tuple))
            else:
                partitioning_tuple = partitioning_tuple[:len(model.layers)]
    
    # Track the most recent computational layer's partitioning
    last_computational_partitioning = None
    computational_layers = (layers.Conv1D, layers.Conv2D, layers.Conv3D, layers.Dense, layers.DepthwiseConv2D, layers.DepthwiseConv1D, layers.SeparableConv2D)
    
    layers_skip = (layers.Flatten, layers.Reshape, layers.Add, layers.Identity)
    
    # first run for the input layer
    input_layer = model.layers[0]
    
    # Get partitioning for input layer (first element in the list)
    spat, out_ch, in_ch = partitioning_tuple[0]
    partitions[input_layer.name] = _build_partitions_from_layer(input_layer, spat, out_ch, in_ch)
    
    if verbose:
        print(f"Layer {input_layer.name} partitioned successfully with partition parameters: {(spat, out_ch, in_ch)}")

    # Create a mapping from layer to its partitioning tuple
    layer_partitioning = {}
    for i, layer in enumerate(model.layers):
        layer_partitioning[layer.name] = partitioning_tuple[i]
    
    # Track which layers have been partitioned
    partitioned_layers = set()
    # Track which output layers have been grouped (to avoid re-marking their partitions)
    # For multi-input layers, the first input will do full grouping with marking,
    # subsequent inputs will only add dependencies without marking
    grouped_output_layers = set()

    # then proceed with the other layers
    for _, prev_layer, layer in sorted(layer_deps, key=lambda x: x[0]):

        # if the layer is in a skipped layers list we can directly skip it
        if isinstance(layer, layers_skip):
            print(f"Skipping layer {layer.name}")
            print(f"Warning: Layer {layer.name} is skipped, might be a problem with dependencies.")
            continue

        # Create partitions for this layer (only once)
        if layer.name not in partitioned_layers:
            # Get partitioning for this specific layer
            spat, out_ch, in_ch = layer_partitioning[layer.name]
            partitions[layer.name] = _build_partitions_from_layer(layer, spat, out_ch, in_ch)
            partitioned_layers.add(layer.name)

            # Update tracking of computational layers
            if isinstance(layer, computational_layers):
                last_computational_partitioning = (spat, out_ch, in_ch)
                if verbose:
                    print(f"Updated computational layer tracking: {last_computational_partitioning}")

            if verbose:
                print(f"Layer {layer.name} partitioned successfully with partition parameters: {(spat, out_ch, in_ch)}")

        # Group partitions for this dependency edge
        # This handles multi-input layers (e.g., after ADD) by processing each input separately
        if prev_layer.name in partitions:
            # Check if this output layer has already been grouped
            if layer.name not in grouped_output_layers:
                # First time seeing this output layer - do full grouping with marking
                partitions1 = partitions[prev_layer.name]
                partitions2 = partitions[layer.name]
                _group_partitions(partitions1, partitions2, layer_deps, partitions_deps)
                grouped_output_layers.add(layer.name)
            else:
                # Multi-input layer: just build dependencies without re-marking partitions
                partitions1 = partitions[prev_layer.name]
                partitions2 = partitions[layer.name]
                # Build dependencies only (skip the marking done in _group_partitions)
                temp_deps = _build_partitions_deps(partitions1, partitions2, layer_deps)
                name_layer1 = partitions1[0].id.split('-')[0]
                name_layer2 = partitions2[0].id.split('-')[0]
                if (name_layer1, name_layer2) in temp_deps:
                    # Add to existing dependencies or create new entry
                    if (name_layer1, name_layer2) in partitions_deps:
                        # Merge with existing dependencies
                        for key, value in temp_deps[(name_layer1, name_layer2)].items():
                            partitions_deps[(name_layer1, name_layer2)][key] = value
                    else:
                        partitions_deps[(name_layer1, name_layer2)] = temp_deps[(name_layer1, name_layer2)]

    if grouping:
        _section_partitions(partitions, partitions_deps, pe.mem_size)
        partitions, partitions_deps = _build_straight_through_deps(partitions=partitions, partitions_deps=partitions_deps)

    return partitions, partitions_deps

def row_wise_mapping(task_graph, domain, verbose = False):
        """
        Generate a path in a X direction on the grid of PEs.
        
        The path consists of tuples (task_id, current_node, next_node) representing:
        - The task being processed
        - The node where the task starts
        - The node where the task will be executed
    
        
        Returns:
            list: A path represented as a list of (task_id, current_node, next_node) tuples
            
        """
        
        tasks = [task["id"] for task in task_graph.get_nodes()] 
        
        #initilaize the path and resource tracking
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        # A list of the available resources for each PE
        resources = [PE() for _ in range(domain.size)]
        # the last node is decalred as drain point and the starting point is source point
        source_node = task_graph.SOURCE_POINT
        drain_node = task_graph.DRAIN_POINT
        
        #start with previous node as -1 (no previous node yet)
        prev_node = -1
    
        for task_id in tasks:
            current_node = prev_node
            
            #determine resources requirmnets for this task
            if task_id not in ("start", "end"):
                task_size = task_graph.get_node(task_id)["size"]
            else:
                task_size = 0
            
            #Handle special case for start and end tasks
            if task_id == "start":
                next_node = source_node
            #case to map last on the drain node
            
            elif task_id == "end": #case to connect last to "end"
                next_node = drain_node 
            else:
                next_node = task_id % domain.size
                #print(f"Task id: {task_id} and next node {next_node}")
                #np.random.choice(range(domain.size), 1)[0]
            
            # udpate the resources
            if task_id != "start" and task_id != "end":
                resources[next_node].mem_used += task_size
            
            #normal case
            path.append((task_id, current_node, next_node))
            prev_node = next_node

        if verbose:
            print("proposed path is:", path)
        return path

def column_wise_mapping(task_graph, domain, verbose = False):
        """
        Generate a path in a Y direction on the grid of PEs.

        The path consists of tuples (task_id, current_node, next_node) representing:
        - The task being processed
        - The node where the task starts
        - The node where the task will be executed
    
        
        Returns:
            list: A path represented as a list of (task_id, current_node, next_node) tuples
            
        """
        
        tasks = [task["id"] for task in task_graph.get_nodes()] 
        
        #initilaize the path and resource tracking
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        # A list of the available resources for each PE
        resources = [PE() for _ in range(domain.size)]
        # the last node is decalred as drain point and the starting point is source point
        source_node = task_graph.SOURCE_POINT
        drain_node = task_graph.DRAIN_POINT
        
        #start with previous node as -1 (no previous node yet)
        prev_node = -1
    
        for task_id in tasks:
            current_node = prev_node
            
            #determine resources requirmnets for this task
            if task_id not in ("start", "end"):
                task_size = task_graph.get_node(task_id)["size"]
            else:
                task_size = 0
            
            #Handle special case for start and end tasks
            if task_id == "start":
                next_node = source_node
            #case to map last on the drain node
            
            elif task_id == "end": #case to connect last to "end"
                next_node = drain_node 
            else:
                #K is the number of PEs in one dimension, size is the K * N (which N is the number of dimensions)
                rows = domain.size // domain.K
                col = task_id // rows
                row = task_id % rows
                next_node = (row * domain.K + col) % domain.size
                #next_node = (task_id * domain.K) % domain.size
                #print(f"Task id: {task_id} and next node {next_node}")
                #np.random.choice(range(domain.size), 1)[0]
            
            # udpate the resources
            if task_id != "start" and task_id != "end":
                resources[next_node].mem_used += task_size
            
            #normal case
            path.append((task_id, current_node, next_node))
            prev_node = next_node

        if verbose:
            print("proposed path is:", path)
        return path
    
def random_mapping(task_graph, domain, verbose = False):
        """
        Generate a random path on the grid of PEs.
        
        The path consists of tuples (task_id, current_node, next_node) representing:
        - The task being processed
        - The node where the task starts
        - The node where the task will be executed
    
        
        Returns:
            list: A path represented as a list of (task_id, current_node, next_node) tuples
            
        """
        
        tasks = [task["id"] for task in task_graph.get_nodes()] 
        
        #initilaize the path and resource tracking
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        # A list of the available resources for each PE
        resources = [PE() for _ in range(domain.size)]
        # the last node is decalred as drain point and the starting point is source point
        source_node = task_graph.SOURCE_POINT
        drain_node = task_graph.DRAIN_POINT
        
        #start with previous node as -1 (no previous node yet)
        prev_node = -1
    
        for task_id in tasks:
            current_node = prev_node
            
            #determine resources requirmnets for this task
            if task_id not in ("start", "end"):
                task_size = task_graph.get_node(task_id)["size"]
            else:
                task_size = 0
            
            #Handle special case for start and end tasks
            if task_id == "start":
                next_node = source_node
            #case to map last on the drain node
            
            elif task_id == "end": #case to connect last to "end"
                next_node = drain_node 
            else:
                #choose a random PE from the domain
                next_node = np.random.choice(range(domain.size))
                #print(f"Task id: {task_id} and next node {next_node}")
                #np.random.choice(range(domain.size), 1)[0]
            
            # udpate the resources
            if task_id != "start" and task_id != "end":
                resources[next_node].mem_used += task_size
            
            #normal case
            path.append((task_id, current_node, next_node))
            prev_node = next_node

        if verbose:
            print("proposed path is:", path)
        return path

"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
3. Search space exploration for partitioning functions
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
"""

def split_factor_only_one_strategy(layer, strategy = "spatial", factor=10, path="data" + "/partitioner_data", model_name = "test"):
        """
        Explores partitioning combinations for a single strategy (spatial, output, or input).
        Only varies the specified strategy dimension while keeping others at 1.
        """
        print("\n====================================================")
        print(f"Exploring {strategy} partitioning combinations for layer {layer.name}")
        if strategy == "spatial":
            print(f"Maximum splitting factors: {factor}x{1}x{1}")
        elif strategy == "output":
            print(f"Maximum splitting factors: {0}x{factor}x{1}")
        elif strategy == "input":
            print(f"Maximum splitting factors: {0}x{1}x{factor}")
        print("====================================================\n")


        # Configuration
        max_splitting_factor = factor
        computational_layers = (layers.Conv2D, layers.Dense, layers.DepthwiseConv2D, layers.SeparableConv2D)
        auxiliary_layers = (
            layers.BatchNormalization,
            layers.MaxPooling2D,
            layers.AveragePooling2D,
            layers.GlobalMaxPooling2D,
            layers.GlobalAveragePooling2D,
            layers.Dropout,
            layers.Activation,
            layers.ReLU,
            layers.LeakyReLU
        )

        # Special case: InputLayer
        if isinstance(layer, layers.InputLayer):
            print("InputLayer detected - no partitioning needed")
            return 0, 1, 1

        # Classify layer type
        is_computational = isinstance(layer, computational_layers)
        is_auxiliary = isinstance(layer, auxiliary_layers)

        print(f"Layer classification - Computational: {is_computational}, Auxiliary: {is_auxiliary}")

        if not is_computational and not is_auxiliary:
            print("WARNING: Layer is neither computational nor auxiliary")
            return 0, 1, 1

        # Data collection structure
        results = []
        iteration = 0

        # Define ranges based on strategy - only vary one dimension
        if strategy == "spatial":
            spatial_range = range(1, max_splitting_factor + 1)
            output_range = [1]  # Fixed at 1
            input_range = [1]   # Fixed at 1
        elif strategy == "output":
            spatial_range = [0]  # Fixed at 0
            output_range = range(1, max_splitting_factor + 1)
            input_range = [1]    # Fixed at 1
        elif strategy == "input":
            spatial_range = [0]  # Fixed at 0
            output_range = [1]   # Fixed at 1
            input_range = range(1, max_splitting_factor + 1)
        else:
            print(f"Unknown strategy: {strategy}. Using spatial as default.")
            raise ValueError("Invalid strategy specified.")

        # Iterate through combinations based on strategy
        for spatial in spatial_range:
            for output in output_range:
                for input_split in input_range:
                    try:
                        # Build partitions with current splitting factors
                        partitions = _build_partitions_from_layer(layer, spatial, output, input_split)

                        # Calculate FLOP metrics
                        max_flops = max([p.FLOPs for p in partitions]) if partitions else 0
                        avg_flops = sum([p.FLOPs for p in partitions]) / len(partitions) if partitions else 0
                        #inputs + output + kernels size of the partitions
                        max_size_partition = max([p.tot_size for p in partitions]) if partitions else 0
                        avg_size_partition = sum([p.tot_size for p in partitions]) / len(partitions) if partitions else 0

                        # Store results
                        results.append({
                            'iteration': iteration,
                            'max_flops': max_flops,
                            'avg_flops': avg_flops,
                            'max_size': max_size_partition,
                            'avg_size': avg_size_partition,
                            'spatial': spatial,
                            'output': output,
                            'input': input_split,
                            'valid': True,
                            'partitions': len(partitions)
                        })

                        print(f"Iteration {iteration}: S:{spatial}, O:{output}, I:{input_split} | "
                            f"Max FLOPs: {max_flops:.0f}, Avg: {avg_flops:.0f} | Partitions: {len(partitions)} | Valid: {True}")

                        iteration += 1

                    except ValueError as e:
                        # Store invalid combinations
                        results.append({
                            'iteration': iteration,
                            'max_flops': 0,
                            'avg_flops': 0,
                            'max_size': 0,
                            'avg_size': 0,
                            'spatial': spatial,
                            'output': output,
                            'input': input_split,
                            'valid': False,
                            'error': str(e),
                            'partitions': 0
                        })
                        print(f"Iteration {iteration}: S:{spatial}, O:{output}, I:{input_split} | INVALID: {str(e)}")
                        iteration += 1

        # Save results to CSV
        if results:
            df = pd.DataFrame(results)

            # Define the combined CSV filename
            csv_filename = f"{path}/{model_name}_parts_explo_all_layers_{strategy}.csv"

            # Check if file exists to determine if header is needed
            file_exists = os.path.exists(csv_filename)

            # Create CSV data with layer name column
            if not file_exists:
                # Write header if file doesn't exist
                csv_data = "Layer_Name,Iteration,Max_FLOPs,Avg_FLOPs,Max_Size,Avg_Size,Spatial,Output,Input,Valid,Partitions\n"
            else:
                csv_data = ""

            # Add data rows with layer name
            for row in results:
                csv_data += f"{layer.name},{row['iteration']},{row['max_flops']},{row['avg_flops']},{row['max_size']},{row['avg_size']},{row['spatial']},{row['output']},{row['input']},{row['valid']},{row['partitions']}\n"

            # Append to file (or create if doesn't exist)
            mode = "a" if file_exists else "w"
            with open(csv_filename, mode) as f:
                f.write(csv_data)

            print(f"Appended partitioning results for layer {layer.name} to: {csv_filename}")
            
            return 1, 1, 1
        else:
            print("No valid partitioning found, using minimal partitioning (1, 1, 1)")
            return 1, 1, 1

def search_space_split_factors(layer, factor=10, FLOP_threshold=1e9, size_of_grid = 16, return_best_valid=True, path = "data" + "/partitioner_data"):
    """
    Explores ALL possible partitioning combinations (up to factor x factor x factor)
    and collects FLOPs data for valid configurations.
    """
    print("\n====================================================")
    print(f"Exploring ALL partitioning combinations for layer {layer.name}")
    print(f"Maximum splitting factors: {factor}x{factor}x{factor}")
    
    # Configuration
    max_splitting_factor = factor
    computational_layers = (layers.Conv2D, layers.Dense, layers.DepthwiseConv2D, layers.SeparableConv2D)
    auxiliary_layers = (
        layers.BatchNormalization, 
        layers.MaxPooling2D, 
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
        layers.GlobalAveragePooling2D,
        layers.Dropout,
        layers.Activation,
        layers.ReLU,
        layers.LeakyReLU
    )
    
    # Special case: InputLayer
    if isinstance(layer, layers.InputLayer):
        print("InputLayer detected - no partitioning needed")
        return 0, 1, 1
    
    # Classify layer type
    is_computational = isinstance(layer, computational_layers)
    is_auxiliary = isinstance(layer, auxiliary_layers)
    
    print(f"Layer classification - Computational: {is_computational}, Auxiliary: {is_auxiliary}")
    
    if not is_computational and not is_auxiliary:
        print("WARNING: Layer is neither computational nor auxiliary")
        return 0, 1, 1

    # Data collection structure
    results = []
    iteration = 0
    
    # For computational layers, explore all combinations
    # Iterate through all possible combinations
    for spatial in range(1, max_splitting_factor + 1):
        for output in range(1, max_splitting_factor + 1):
            for input_split in range(1, max_splitting_factor + 1):
                try:
                    # Build partitions with current splitting factors
                    partitions = _build_partitions_from_layer(layer, spatial, output, input_split)

                    # Calculate FLOP metrics
                    max_flops = max([p.FLOPs for p in partitions]) if partitions else 0
                    avg_flops = sum([p.FLOPs for p in partitions]) / len(partitions) if partitions else 0
                    #inputs + output + kernels size of the partitions
                    max_size_partition = max([p.tot_size for p in partitions]) if partitions else 0
                    avg_size_partition = sum([p.tot_size for p in partitions]) / len(partitions) if partitions else 0
                    
                    # Check if number of partitions is valid
                    is_valid = len(partitions) >= size_of_grid

                    # Store results
                    results.append({
                        'iteration': iteration,
                        'max_flops': max_flops,
                        'avg_flops': avg_flops,
                        'max_size': max_size_partition,
                        'avg_size': avg_size_partition,
                        'spatial': spatial,
                        'output': output,
                        'input': input_split,
                        'valid': is_valid,
                        'partitions': len(partitions)
                    })
                    
                    print(f"Iteration {iteration}: S:{spatial}, O:{output}, I:{input_split} | "
                        f"Max FLOPs: {max_flops:.0f}, Avg: {avg_flops:.0f} | Partitions: {len(partitions)} | Valid: {is_valid}")
                    
                    iteration += 1
                    
                except ValueError as e:
                    # Store invalid combinations
                    results.append({
                        'iteration': iteration,
                        'max_flops': 0,
                        'avg_flops': 0,
                        'max_size': 0,
                        'avg_size': 0,
                        'spatial': spatial,
                        'output': output,
                        'input': input_split,
                        'valid': False,
                        'error': str(e),
                        'partitions': 0
                    })
                    print(f"Iteration {iteration}: S:{spatial}, O:{output}, I:{input_split} | INVALID: {str(e)}")
                    iteration += 1
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Create CSV string
        csv_data = "Iteration,Max_FLOPs,Avg_FLOPs,Max_Size,Avg_Size,Spatial,Output,Kernel,Valid,Partitions\n"
        for row in results:
            csv_data += f"{row['iteration']},{row['max_flops']},{row['avg_flops']},{row['max_size']},{row['avg_size']},{row['spatial']},{row['output']},{row['input']},{row['valid']},{row['partitions']}\n"
        
        # Save to file
        with open(path + f"/parts_explo_{layer.name}.csv", "w") as f:
            f.write(csv_data)
        
        print(f"path + /parts_explo_{layer.name}.csv")
    
    # Return the best valid partitioning (lowest max FLOPs that meets threshold)
    if len(results) > 0:
        if return_best_valid == True:
            
            valid_results = [r for r in results if r['valid']]
            if valid_results:
                
                # Sort by max FLOPs (ascending) and total partitions (descending)
                valid_results.sort(key=lambda x: (x['max_flops'], -(x['spatial'] * x['output'] * x['input'])))
                
                # Find the first result that meets the threshold
                for result in valid_results:
                    if result['max_flops'] <= FLOP_threshold:
                        best_result = result
                        break
                else:
                    # If none meet threshold, use the one with lowest max FLOPs
                    best_result = valid_results[0]
                
                print("\nBest valid partitioning found:")
                print(f"Spatial: {best_result['spatial']}, Output: {best_result['output']}, Input: {best_result['input']}")
                print(f"Max FLOPs: {best_result['max_flops']:.0f}, Avg FLOPs: {best_result['avg_flops']:.0f}")
                print(f"Max Size: {best_result['max_size']:.0f}, Avg Size: {best_result['avg_size']:.0f}")
                print(f"Partitions: {best_result['partitions']}")
                
                return best_result['spatial'], best_result['output'], best_result['input']
        else:
            print(f"\nReturning last valid partitioning found: Spatial: {results[-1]['spatial']}, Output: {results[-1]['output']}, Input: {results[-1]['input']}")
            return results[-1]['spatial'], results[-1]['output'], results[-1]['input']
    else:
        print("No valid partitioning found, using minimal partitioning (1, 1, 1)")
        return 1, 1, 1


"""
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
4. MODEL ANALYSIS FUNCTIONS: used to analyze the model and compute the FLOPs and MACs.
Remember each hook has order of return: FLOPs, MACs! 
* = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = * = *
"""

def analyze_partition(partition):
    """
    The function gets as input a PartitionInfo object and computes the number of FLOPs (and MACs if available) needed to perform the computation for a single sample

    Args:
    - partition : the partition for which we want to compute the FLOPs (and MACs)

    Returns:
    - FLOPs : the number of FLOPs performed in the partition
    """

    FLOPs = 0
    MACs = 0
    tot_par_size = 0
    precision = 2 # bytes per parameter (assuming float16) 1 is 1 byte (int8), float32 is 4 bytes, float64 is 8 bytes

    if isinstance(partition.layer, (layers.InputLayer, layers.Reshape, layers.ZeroPadding1D, layers.ZeroPadding2D, layers.Identity, layers.Flatten)):
        return MACs, FLOPs, tot_par_size

    # Get the layer and the input and output shapes of the partition
    layer = partition.layer
    inputs_bounds = partition.in_bounds
    inputs_shape =[inputs_bounds[1][i] - inputs_bounds[0][i] for i in range(len(inputs_bounds[0]))] if len(inputs_bounds[0]) > 1 else [inputs_bounds[1][0] - inputs_bounds[0][0]]
    tot_par_size += np.prod(inputs_shape) * ((partition.in_ch[1] - partition.in_ch[0]) if partition.in_ch is not None else 1) * precision
    # prepend a 0 to the input shape to make it compatible to the hooks
    inputs_shape = [0] + inputs_shape + ([partition.in_ch[1] - partition.in_ch[0]] if partition.in_ch is not None else [1])
    outputs_bounds = partition.out_bounds
    outputs_shape = [outputs_bounds[1][i] - outputs_bounds[0][i] for i in range(len(outputs_bounds[0]))] if len(outputs_bounds[0]) > 1 else [outputs_bounds[1][0] - outputs_bounds[0][0]]
    tot_par_size += np.prod(outputs_shape) * ((partition.out_ch[1] - partition.out_ch[0]) if partition.out_ch is not None else 1) * precision
    # prepend a 0 to the output shape also
    outputs_shape = [0] + outputs_shape + ([partition.out_ch[1] - partition.out_ch[0]] if partition.out_ch is not None else [1])
    # Compute the FLOPs (and MACs) using the hook
    if type(layer) in register_hooks:
        FLOPs, MACs = register_hooks[type(layer)](layer, inputs_shape, outputs_shape)
        # if the partitioned layer also has an activation function, we append also those FLOPs and MACs
        if hasattr(layer, "activation"):
            activation = layer.activation.__name__
            FLOPs_act, MACs_act = register_hooks[activation](layer, inputs_shape, outputs_shape)
            MACs += MACs_act
            FLOPs += FLOPs_act

    for weight in partition.weights_shape:
        tot_par_size += np.prod(weight) * precision

    return FLOPs, MACs, tot_par_size

def _analyze_layer(layer,activation = None):
    '''
    The function gets as input a keras layer and computes the number of FLOPs (and MACs if available) needed to perform the computation for a single sample.

    Parameters:
    - layer : the layer for which we want to compute the FLOPs (and MACs)
    - activation : the activation function of the layer (if available)

    Returns:
    - FLOPs : the number of FLOPs performed in the layer
    - MACs : the number of MACs performed in the layer (if available)
    '''
    FLOPs = 0
    MACs = 0

    # We have to possiblityes: either the layer is a keras layer or an activation function
    if activation is not None:
        hook = register_hooks[activation]
    else:
        hook = register_hooks[type(layer)]

    if hook is not None:
        # Get the input and output shapes of the layer
        inputs_shape = layer.input_shape
        outputs_shape = layer.output_shape
        # Compute the FLOPs (and MACs) using the hook
        FLOPs, MACs = hook(layer, inputs_shape, outputs_shape)

    return FLOPs, MACs 


def analyze_ops(model: keras.Model, incl_info = False):
    '''
    The function gets as input a model and computes the number of MACs and FLOPs needed to perform the computation.
    It then prints out a summary of the model, with the number of FLOPs (and MACs, when available) for each layer using the hooks defined above.

    Parameters:
    - model : the model for which we want to compute the MACs/FLOPs
    - include_info : a flag that specifies if we want to include additional information in the summary

    Returns:
    - a table with the number of MACs and FLOPs for each layer of the model
    and the total number of MACs and FLOPs for the whole model
    '''

    print("--------------------------------------------* Model " + model.name + " Parameters *--------------------------------------------")
    total_MACs = 0
    total_FLOPs = 0
    included_info = ['Input parameters (bytes)', 'Input Shape', 'Weights (bytes)', 'Weights Shape', 'Output parameters (bytes)', 'Output Shape'] if incl_info else []

    # Create a PrettyTable instance
    table = pt.PrettyTable()
    table.field_names = ["Layer", "Layer number"] + included_info + [ "FLOPs", "MACs"]
    total_parameters = 0

    # Iterate over layers and activations in the model
    for i, layer in enumerate(model.layers):
        # hook = register_hooks[type(layer)]
        FLOPs, MACs = _analyze_layer(layer)
        total_MACs += MACs
        total_FLOPs += FLOPs

        if incl_info:
            # Get the number of input parameters
            input_params = np.prod(layer.input.shape[1:]) if type(layer.input) != list else 2*np.prod(layer.input[0].shape[1:])
            input_dim = layer.input.shape[1:] if type(layer.input) != list else layer.input[0].shape[1:]
            
            # Get the number of weights
            weights = int(np.sum([np.prod(w.shape) for w in layer.get_weights()]))
            # Take into accont the bias
            if layer in [layers.Conv2D, layers.Conv1D, layers.DepthwiseConv2D, layers.DepthwiseConv1D, layers.Conv2DTranspose, layers.Conv1DTranspose] and layer.use_bias:
                weights += len(layer.get_weights())
            
            weights_dim = [w.shape for w in layer.get_weights()]
            
            # Get the number of output parameters
            output_params = np.prod(layer.output.shape[1:])
            output_dim = layer.output.shape[1:]

            
            # Add the number of weights to the total number of parameters
            total_parameters += weights + output_params + input_params

            table.add_row([layer.name, i, input_params, input_dim, weights, weights_dim, output_params, output_dim, FLOPs, MACs], divider = False if hasattr(layer, "activation") and type(layer) != layers.Activation else True)

            if hasattr(layer, "activation") and type(layer) != layers.Activation:
                activation = layer.activation.__name__
                FLOPs_act, MACs_act = _analyze_layer(layer, activation)
                total_MACs += MACs_act
                total_FLOPs += FLOPs_act
                table.add_row([activation, i,output_params, output_dim, 0, [], output_params, output_dim, FLOPs_act, MACs_act], divider = True)
            
            
        else: 
            # Add a row to the table for this layer
            table.add_row([layer.name, i, FLOPs, MACs], divider = False if hasattr(layer, "activation") and type(layer) != layers.Activation else True)

            # For each layer, also get the activation function if available and add it to the table
            if hasattr(layer, "activation") and type(layer) != layers.Activation:
                activation = layer.activation.__name__
                FLOPs_act, MACs_act = _analyze_layer(layer, activation)
                total_MACs += MACs_act
                total_FLOPs += FLOPs_act
                table.add_row([activation, i, FLOPs_act, MACs_act], divider = True)

            

    print(table)
    print(f"Total parameters: {total_parameters}")
    print(f"Total: MACs={total_MACs}, FLOPs={total_FLOPs}")
    print("------------------------------------------------------------------------------------------------------------------------")
    
def print_partitions_table(partitions: Dict[str, List[PartitionInfo]], partitions_deps: Dict[Tuple[str, str], int]) -> None:
    """
    Enhanced function that prints a readable table with partitions, optimized for terminal width.
    Uses multiple compact tables or vertical layout for better readability.
    
    Args:
    - partitions: a dictionary of partitions of the model
    - partitions_deps: a dictionary of dependencies between the partitions of the model
    """
    
    # Get terminal width, default to 80 if not available
    try:
        terminal_width = os.get_terminal_size().columns
    except:
        terminal_width = 80
    
    # Target width slightly less than terminal to avoid wrapping
    target_width = min(terminal_width - 5, 120)
    
    print("=" * target_width)
    print(f"{'* PARTITIONS SUMMARY *':^{target_width}}")
    print("=" * target_width)
    
    # Option 1: Compact horizontal table
    if target_width >= 100:
        _print_compact_horizontal_table(partitions, target_width)
    else:
        # Option 2: Vertical layout for narrow terminals
        _print_vertical_layout(partitions, target_width)
    
    print("=" * target_width)
        
    # Summary statistics
    _print_partition_summary(partitions, target_width)


def _print_compact_horizontal_table(partitions: Dict[str, List[PartitionInfo]], max_width: int) -> None:
    """Print a compact horizontal table that fits within the specified width."""
    
    table = pt.PrettyTable()
    table.field_names = ["Layer", "ID", "In Bounds", "Out Bounds", "Ch", "FLOPs", "Size(B)"]
    
    # Set column alignment and width
    table.align["Layer"] = "l"
    table.align["ID"] = "c"
    table.align["In Bounds"] = "c"
    table.align["Out Bounds"] = "c"
    table.align["Ch"] = "c"
    table.align["FLOPs"] = "r"
    table.align["Size(B)"] = "r"
    
    # Limit column widths
    table.max_width["Layer"] = 12
    table.max_width["ID"] = 8
    table.max_width["In Bounds"] = 15
    table.max_width["Out Bounds"] = 15
    table.max_width["Ch"] = 8
    
    for layer_name, partitions_list in partitions.items():
        for i, partition in enumerate(partitions_list):
            # Truncate layer name if too long
            short_layer = layer_name[:12] if len(layer_name) > 12 else layer_name
            
            # Simplified partition ID
            partition_id = f"{i}"
            
            # Compact bounds representation
            in_bounds_str = _format_bounds_compact(partition.in_bounds)
            out_bounds_str = _format_bounds_compact(partition.out_bounds)
            
            # Channel info (input->output) - Handle None cases
            def format_channel_info(ch_range):
                if ch_range is None or len(ch_range) < 2:
                    return "N/A"
                return str(ch_range[1] - ch_range[0])
            
            in_ch_str = format_channel_info(partition.in_ch)
            out_ch_str = format_channel_info(partition.out_ch)
            ch_info = f"{in_ch_str}→{out_ch_str}"
            
            # Format large numbers
            flops_str = _format_number(partition.FLOPs)
            size_str = _format_number(partition.tot_size)
            
            table.add_row([short_layer, partition_id, in_bounds_str, out_bounds_str, 
                        ch_info, flops_str, size_str])
    
    print(table)


def _print_vertical_layout(partitions: Dict[str, List[PartitionInfo]], max_width: int) -> None:
    """Print partitions in vertical layout for narrow terminals."""
    
    for layer_name, partitions_list in partitions.items():
        print(f"\n📋 Layer: {layer_name}")
        print("-" * min(max_width, 50))
        
        for i, partition in enumerate(partitions_list):
            print(f"  Partition {i}:")
            print(f"    Input:  {_format_bounds_compact(partition.in_bounds)}")
            print(f"    Output: {_format_bounds_compact(partition.out_bounds)}")
            print(f"    Channels: {partition.in_ch[1]-partition.in_ch[0] if partition.in_ch else 'N/A'} → {partition.out_ch[1]-partition.out_ch[0] if partition.out_ch else 'N/A'}") 
            print(f"    FLOPs: {_format_number(partition.FLOPs)}")
            print(f"    Size: {_format_number(partition.tot_size)} bytes")
            if i < len(partitions_list) - 1:
                print()


def _format_bounds_compact(bounds) -> str:
    """Format bounds in a compact way."""
    if not bounds or len(bounds) == 0:
        return "N/A"
    
    # Handle case where inner tuples have only 1 element
    def format_single_bound(bound):
        if len(bound) >= 2:
            return f"{bound[0]}:{bound[1]}"
        else:  # Only 1 element (e.g., (0,) or [300])
            return f"{bound[0]}:{bound[0]}"  # Repeat the same value
    
    if len(bounds) == 1:
        return f"({format_single_bound(bounds[0])})"
    elif len(bounds) == 2:
        return f"({format_single_bound(bounds[0])},{format_single_bound(bounds[1])})"
    else:
        # For higher dimensions, show only first two
        return f"({format_single_bound(bounds[0])},{format_single_bound(bounds[1])}...)"


def _format_number(num) -> str:
    """Format numbers in a compact, readable way."""
    if num == 0:
        return "0"
    elif num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}G"


def _print_partition_summary(partitions: Dict[str, List[PartitionInfo]], max_width: int) -> None:
    """Print summary statistics about the partitions."""
    
    total_partitions = sum(len(partitions_list) for partitions_list in partitions.values())
    total_flops = sum(partition.FLOPs for partitions_list in partitions.values() 
                    for partition in partitions_list)
    total_size = sum(partition.tot_size for partitions_list in partitions.values() 
                    for partition in partitions_list)
    
    print(f"\n📊 Summary:")
    print(f"  Total Partitions: {total_partitions}")
    print(f"  Total FLOPs: {_format_number(total_flops)}")
    print(f"  Total Size: {_format_number(total_size)} bytes")
    
    # Per-layer breakdown
    print(f"\n📈 Per-layer breakdown:")
    for layer_name, partitions_list in partitions.items():
        layer_partitions = len(partitions_list)
        layer_flops = sum(p.FLOPs for p in partitions_list)
        layer_size = sum(p.tot_size for p in partitions_list)
        
        print(f"  {layer_name[:15]:<15}: {layer_partitions:2d} parts, "
            f"{_format_number(layer_flops):>8} FLOPs, {_format_number(layer_size):>8}B")


# Alternative: Super compact single-line per partition
def print_partitions_compact(partitions: Dict[str, List[PartitionInfo]]) -> None:
    """Ultra-compact version - one line per partition."""
    
    print("🔧 Compact Partition View:")
    print("-" * 60)
    
    for layer_name, partitions_list in partitions.items():
        short_name = layer_name[:10]
        for i, p in enumerate(partitions_list):
            bounds = _format_bounds_compact(p.in_bounds)
            flops = _format_number(p.FLOPs)
            size = _format_number(p.tot_size)
            print(f"{short_name}[{i}]: {bounds} → {flops} FLOPs, {size}B")


# Example usage with different display modes
def print_partitions_table_adaptive(partitions: Dict[str, List[PartitionInfo]], partitions_deps: Dict[Tuple[str, str], int], mode: str = "auto") -> None:
    """
    Adaptive partition printing with multiple display modes.
    
    Args:
    - partitions: partition dictionary
    - partitions_deps: dependencies dictionary  
    - mode: "auto", "compact", "vertical", or "minimal"
    """
    
    if mode == "auto":
        print_partitions_table(partitions, partitions_deps)
    elif mode == "compact":
        print_partitions_compact(partitions)
    elif mode == "vertical":
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 80
        _print_vertical_layout(partitions, terminal_width)
    elif mode == "minimal":
        # Just show counts and totals
        total_parts = sum(len(p_list) for p_list in partitions.values())
        total_flops = sum(p.FLOPs for p_list in partitions.values() for p in p_list)
        print(f"📊 {len(partitions)} layers, {total_parts} partitions, {_format_number(total_flops)} total FLOPs")
    else:
        print(f"Unknown mode: {mode}. Using auto mode.")
        print_partitions_table(partitions, partitions_deps)
