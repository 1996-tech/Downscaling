import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Activation, BatchNormalization, 
    Dropout, Add
)
from typing import List, Optional, Union

def hidden_block(
    inlayer: layers.Layer,
    nodes: List[int],
    acts: List[str],
    idepth: int,
    orginlayer: Optional[layers.Layer] = None,
    reg: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
    dropout: float = 0.0,
    batchnorm: bool = True
) -> layers.Layer:
    """
    Construct the hidden block for neural network layers.
    
    Parameters:
    - inlayer: input layer (Keras layer)
    - nodes: list of integers, number of nodes for all hidden layers
    - acts: list of strings, activation functions for hidden layers
    - idepth: integer, index of the hidden layer
    - orginlayer: Keras layer, original layer to be added to decoding layer (default: None)
    - reg: regularization (default: None)
    - dropout: dropout rate for the target hidden layer (default: 0.0)
    - batchnorm: flag to conduct batch normalization (default: True)
    
    Returns:
    - Keras layer: block of a hidden layer with activation and/or batch normalization
    """
    
    # Create dense layer
    layer = Dense(
        units=nodes[idepth],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=reg
    )(inlayer)
    
    # Add activation
    layer = Activation(acts[idepth])(layer)
    
    # Add batch normalization if specified
    if batchnorm:
        layer = BatchNormalization()(layer)
    
    # Add dropout if specified
    if dropout is not None and dropout != 0:
        layer = Dropout(dropout)(layer)
    
    # Add residual connection if original layer is provided
    if orginlayer is not None:
        layer = Add()([orginlayer, layer])
        layer = Activation(acts[idepth])(layer)
        if batchnorm:
            layer = BatchNormalization()(layer)
    
    return layer


def AutoEncoderModel(
    nfea: int,
    nout: int,
    nodes: List[int],
    acts: List[str],
    mdropout: float = 0.0,
    reg: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
    batchnorm: bool = True,
    isres: bool = True,
    outtype: int = 0,
    fact: str = "linear"
) -> Model:
    """
    Construct a residual autoencoder-based deep network.
    
    Parameters:
    - nfea: integer, number of features
    - nout: integer, number of output units
    - nodes: list of integers, number of nodes for hidden layers in encoding component
    - acts: list of strings, activation function names
    - mdropout: dropout rate of the coding (middle) layer (default: 0.0)
    - reg: regularization (default: None)
    - batchnorm: flag to conduct batch normalization (default: True)
    - isres: flag to conduct residual connections (default: True)
    - outtype: integer, output type (0: nout outputs, 1: nout+nfea outputs) (default: 0)
    - fact: activation for output layer (default: "linear")
    
    Returns:
    - Keras model: model of (residual) autoencoder-based deep network
    """
    
    # Stack to store encoding layers for skip connections
    encoding_layers = []
    
    # Input layer
    inlayer = Input(shape=(nfea,), name='feat')
    layer = inlayer
    
    # Encoding layers
    for i in range(len(nodes)):
        layer = Dense(
            units=nodes[i],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=reg
        )(layer)
        
        layer = Activation(acts[i])(layer)
        
        if batchnorm:
            layer = BatchNormalization()(layer)
        
        # Store layer for skip connections (except the last encoding layer)
        if i < len(nodes) - 1:
            encoding_layers.append(layer)
        else:
            # Apply dropout to the middle (coding) layer
            if mdropout is not None and mdropout != 0:
                layer = Dropout(mdropout)(layer)
    
    # Decoding layers (in reverse order)
    for i in range(len(nodes) - 2, -1, -1):  # From second-to-last to first
        # Pop the corresponding encoding layer for skip connection
        originlayer = encoding_layers.pop()  # Remove and return the last element
        
        # Use hidden_block function for decoding with skip connections
        layer = hidden_block(
            inlayer=layer,
            nodes=nodes,
            acts=acts,
            idepth=i,
            orginlayer=originlayer,
            reg=reg,
            dropout=None,  # No dropout in decoding layers
            batchnorm=batchnorm
        )
    
    # Reconstruction layer (back to original feature space)
    layer = Dense(
        units=nfea,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=reg
    )(layer)
    
    # Add residual connection with input if specified
    if isres:
        layer = Add()([inlayer, layer])
        layer = Activation(acts[0])(layer)
        if batchnorm:
            layer = BatchNormalization()(layer)
    
    # Determine output size based on outtype
    pnout = nout
    if outtype == 1:
        pnout = pnout + nfea
    
    # Output layer
    outlayer = Dense(
        units=pnout,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation=fact,
        kernel_regularizer=reg
    )(layer)
    
    # Create and return the model
    model = Model(inputs=inlayer, outputs=outlayer)
    
    return model


# Example usage and helper functions
def create_l2_regularizer(l2_factor: float = 0.01):
    """Create L2 regularizer"""
    return tf.keras.regularizers.l2(l2_factor)


def create_l1_regularizer(l1_factor: float = 0.01):
    """Create L1 regularizer"""
    return tf.keras.regularizers.l1(l1_factor)


def create_l1_l2_regularizer(l1_factor: float = 0.01, l2_factor: float = 0.01):
    """Create L1+L2 regularizer"""
    return tf.keras.regularizers.l1_l2(l1=l1_factor, l2=l2_factor)


# Example usage:
if __name__ == "__main__":
    # Example parameters
    nfea = 64  # Number of input features
    nout = 1   # Number of output units
    nodes = [32, 16, 8, 4]  # Hidden layer sizes
    acts = ['relu', 'relu', 'relu', 'relu']  # Activation functions
    mdropout = 0.2  # Middle layer dropout
    reg = create_l2_regularizer(0.01)  # L2 regularization
    batchnorm = True
    isres = True  # Use residual connections
    outtype = 0   # Standard output
    fact = 'linear'  # Output activation
    
    # Create the model
    model = AutoEncoderModel(
        nfea=nfea,
        nout=nout,
        nodes=nodes,
        acts=acts,
        mdropout=mdropout,
        reg=reg,
        batchnorm=batchnorm,
        isres=isres,
        outtype=outtype,
        fact=fact
    )
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Print model summary
    print("Model Summary:")
    model.summary()
    
    # Example of creating a simple hidden block
    input_layer = Input(shape=(10,))
    hidden_layer = hidden_block(
        inlayer=input_layer,
        nodes=[64, 32, 16],
        acts=['relu', 'relu', 'relu'],
        idepth=0,  # First layer
        dropout=0.1,
        batchnorm=True
    )
    
    print("\nHidden block created successfully!")
