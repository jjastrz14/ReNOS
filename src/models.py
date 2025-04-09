import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
import larq


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


def test_conv(input_shape, verbose = False):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (inputs)
    #x = layers.Dense(3, activation='softmax')(x)
    x = layers.Conv2D(4, kernel_size=(3, 3), data_format="channels_last", activation=None) (x)
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