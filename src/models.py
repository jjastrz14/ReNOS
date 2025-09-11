import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import larq


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
    # Last ResBlock (with 64 filters)
    x = ResBlock(inputs, filters=512)
    # Global average pooling and output layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if verbose:
        model.summary()
        print(f'shape of model: {x.shape}')
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
    # First convolution in residual block
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    y = layers.BatchNormalization()(y)
    # Second convolution in residual block
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    # Skip connection (no Add, just pass y and skip connection separately)
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)
    # Output is just the skip connection (x) or y, not added together
    return y  # or return x if you want to preserve only the skip connection

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