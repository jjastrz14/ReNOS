from warnings import filters
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import larq


def AlexNet(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    """
    Lightweight AlexNet adapted for CIFAR-10 (32x32 images).

    Original AlexNet uses 227x227 inputs. This version:
    - Reduces kernel sizes and strides for smaller input
    - Uses BatchNormalization instead of LRN
    - Reduces Dense layer size from 4096 to 1024
    - Adds Dropout for regularization

    Parameters:
        input_shape: tuple, default (32, 32, 3) for CIFAR-10
        num_classes: int, number of output classes
        verbose: bool, print model summary and stats

    Returns:
        keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # Conv Block 1: 32x32x3 -> 16x16x64
    x = layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                        padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)

    # Conv Block 2: 16x16x64 -> 8x8x192
    x = layers.Conv2D(192, kernel_size=(5, 5), padding='same',
                        activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)

    # Conv Block 3: 8x8x192 -> 8x8x384
    x = layers.Conv2D(384, kernel_size=(3, 3), padding='same',
                        activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Block 4: 8x8x384 -> 8x8x256
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Block 5: 8x8x256 -> 4x4x256
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)

    # Flatten: 4x4x256 = 4096 features
    #x = layers.Flatten()(x)

    # FC Block 1: 4096 -> 1024 (reduced from original 4096)
    x = layers.Dense(1024, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)

    # FC Block 2: 1024 -> 1024 (reduced from original 4096)
    x = layers.Dense(1024, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)

    # Output Layer: 1024 -> num_classes
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    if verbose:
        model.summary()
        print(f'Output shape: {outputs.shape}')
        try:
            larq.models.summary(model, print_fn=None, include_macs=True)
        except Exception as e:
            print(f"Could not generate larq summary: {e}")
    return model

def VGG_16_early_layers(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    """
    First 8 convolutional layers of VGG16 (first 3 blocks).

    VGG16 structure:
    - Block 1: 2x Conv(64) + MaxPool  → 32x32 -> 16x16
    - Block 2: 2x Conv(128) + MaxPool → 16x16 -> 8x8
    - Block 3: 3x Conv(256) + MaxPool → 8x8 -> 4x4
    Total: 7 conv layers (stops before the last conv in block 3)

    Output: (4, 4, 256)
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: 32x32x3 -> 16x16x64
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2: 16x16x64 -> 8x8x128
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3: 8x8x128 -> 4x4x256
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    model = keras.Model(inputs=inputs, outputs=x)

    if verbose:
        model.summary()
        print(f'Output shape of early layers: {x.shape}')
        try:
            larq.models.summary(model, print_fn=None, include_macs=True)
        except Exception as e:
            print(f"Could not generate larq summary: {e}")

    return model

def VGG_16_late_layers(input_shape=(4, 4, 256), num_classes=10, verbose=False):
    """
    Last 6 convolutional layers + FC layers of VGG16 (last 2 blocks + classifier).

    VGG16 structure:
    - Block 4: 3x Conv(512) + MaxPool → 4x4 -> 2x2
    - Block 5: 3x Conv(512) + MaxPool → 2x2 -> 1x1
    - FC layers: Flatten -> Dense(4096) -> Dense(4096) -> Dense(num_classes)
    Total: 6 conv layers + 3 FC layers

    Input: (4, 4, 256) from early layers
    Output: (num_classes,)
    """
    inputs = layers.Input(shape=input_shape)

    # Block 4: 4x4x256 -> 2x2x512
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5: 2x2x512 -> 1x1x512
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classifier: Flatten + FC layers
    #x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    if verbose:
        model.summary()
        print(f'Output shape: {outputs.shape}')
        try:
            larq.models.summary(model, print_fn=None, include_macs=True)
        except Exception as e:
            print(f"Could not generate larq summary: {e}")

    return model


def ResNet32_early_blocks(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    """
    Early blocks of ResNet32: initial layers and first stage
    """
    inputs = layers.Input(shape=input_shape)
    
    # First layer: initial convolution
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # First stage: 5 ResBlocks with 16 filters (10 layers total)
    # ResNet32 has 3 stages with 5 ResBlocks each (5*2*3 = 30 layers) + initial conv = 31 layers
    for _ in range(5):
        x = ResBlock(x, filters=16, strides=(1, 1))
    
    model = keras.Model(inputs=inputs, outputs=x)
    
    if verbose:
        model.summary()
        print(f'Output shape of early blocks: {x.shape}')
    
    return model

def ResNet32_mid_blocks(input_shape=(32, 32, 16), num_classes=10, verbose=False):
    """
    Middle blocks of ResNet32: second stage with increased filters
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Second stage: 5 ResBlocks with 32 filters
    # First block uses stride=2 to downsample, others use stride=1
    x = ResBlock(x, filters=32, strides=(2, 2))  # Downsample
    for _ in range(4):
        x = ResBlock(x, filters=32, strides=(1, 1))
    
    model = keras.Model(inputs=inputs, outputs=x)
    
    if verbose:
        model.summary()
        print(f'Output shape of mid blocks: {x.shape}')
    
    return model

def ResNet32_late_blocks(input_shape=(16, 16, 32), num_classes=10, verbose=False):
    """
    Late blocks of ResNet32: third stage and final layers
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Third stage: 5 ResBlocks with 64 filters
    # First block uses stride=2 to downsample, others use stride=1
    x = ResBlock(x, filters=64, strides=(2, 2))  # Downsample
    for _ in range(4):
        x = ResBlock(x, filters=64, strides=(1, 1))
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    
    if verbose:
        model.summary()
        print(f'Output shape before pooling: {x.shape}')
    
    return model



def ResNet_early_blocks(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    inputs = layers.Input(shape=input_shape)
    # First layer: initial convolution
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # First ResBlock (with 16 filters)
    x = ResBlock(x, filters=16)
    model = keras.Model(inputs=inputs, outputs=x)
    if verbose:
        model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    
    return model 

def ResNet_late_blocks(input_shape=(8, 8, 256), num_classes=10, verbose=False):

    inputs = layers.Input(shape=input_shape)

    # Sequential ResBlock (first one)
    x1 = ResBlock(inputs, filters=512)
    # Two parallel ResBlocks (both taking the same input)
    x2 = ResBlock(inputs, filters=512)
    x3 = ResBlock(inputs, filters=512)

    # Create outputs for all branches to ensure they're included in the model
    output1 = layers.Dense(num_classes, activation='softmax', name='output1')(x1)
    output2 = layers.Dense(num_classes, activation='softmax', name='output2')(x2)
    output3 = layers.Dense(num_classes, activation='softmax', name='output3')(x3)

    # Create model with multiple outputs to preserve all branches
    model = keras.Model(inputs=inputs, outputs=[output1, output2, output3])
    if verbose:
        model.summary()
        print(f'shape of model: {x1.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)

    return model 

def VGG_early_layers(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    
    inputs = layers.Input(shape=input_shape)
    # First two layers of VGG16
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    if verbose:
        model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    
    return model

def VGG_late_layers(input_shape=(14, 14, 512), num_classes=10, verbose=False):
    inputs = layers.Input(shape=input_shape)
    # Last two convolutional layers before FC layers in VGG16
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    if verbose:
        model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    
    return model
    

def ResBlock(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """
    Residual block with skip connection.

    Args:
        x: Input tensor
        filters: Number of output filters
        kernel_size: Convolution kernel size
        strides: Convolution strides

    Returns:
        Output tensor after residual connection (x + F(x))
    """
    # Main path: two convolutions with BN
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Skip connection with projection if needed
    shortcut = x
    if x.shape[-1] != filters:
        # Projection shortcut to match dimensions
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    out = layers.Add()([shortcut, y])
    out = layers.ReLU()(out)

    return out

def Resnet9s(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    inputs = layers.Input(shape=input_shape)
    # Conv1
    x = layers.Conv2D(28, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Conv2
    x = layers.Conv2D(28, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool
    x = layers.MaxPooling2D((2, 2))(x)
    # ResBlock3
    x = ResBlock(x, filters=28)
    # Conv4
    x = layers.Conv2D(28, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool
    x = layers.MaxPooling2D((2, 2))(x)
    # Conv5
    x = layers.Conv2D(56, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool
    x = layers.MaxPooling2D((2, 2))(x)
    # ResBlock6
    x = ResBlock(x, filters=56)
    # Global Average Pooling (replaces the 4x4 MaxPool in the table)
    x = layers.GlobalAveragePooling2D()(x)
    # FC Layer (Output layer)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if verbose:
        model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    
    return model

def LeNet4(input_shape=(32, 32, 3), num_classes=10, verbose=False):
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(4, kernel_size=(5, 5), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # Layer 2: Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(12, kernel_size=(5, 5), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # Flatten
    #x = layers.Flatten()(x)
    # Layer 3: FC + BN + ReLU
    x = layers.Dense(100, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if verbose:
        model.summary()        
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def test_model(input_shape, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='linear') (inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, kernel_size=(5, 5))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model


def small_test_model(input_shape, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, kernel_size=(3, 3), activation='linear') (inputs)
    x = layers.Conv2D(3, kernel_size=(3, 3), activation='linear') (x)
    x = layers.Conv2D(4, kernel_size=(3, 3), activation='linear') (x)
    x = layers.Conv2D(4, kernel_size=(3, 3), activation='linear') (x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model


def test_conv(input_shape, num_classes, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(56, kernel_size=(3, 3), data_format="channels_last", activation=None) (x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def single_conv(input_shape, num_classes, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    #x = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def single_conv_big(input_shape, num_classes, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(256, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    #x = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def double_conv(input_shape, num_classes, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (x)
    #x = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def triple_conv(input_shape, num_classes, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (x)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (x)
    #x = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model

def load_model(model_name, verbose = False):
    available_models = ["ResNet50", "MobileNetV2", "MobileNet", "ResNet18"]
    if model_name not in available_models:
        raise ValueError(f"Model not available. Please choose from the following: {', '.join(available_models)}")
    
    # Load the model
    model = keras.applications.__getattribute__(model_name)(weights='imagenet')
    if verbose:
        summary = model.summary()
        print(f'shape of model: {x.shape}')
        larq.models.summary(model, print_fn=None, include_macs=True)
    return model