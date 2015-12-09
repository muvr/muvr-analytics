import numpy as np
import struct
import cPickle as pkl

from neon.initializers import Uniform, Constant
from neon.layers import Affine, Dropout
from neon.transforms import Rectlin, Logistic, Tanh


def extract_weights(file_name):
    """Load weights from the file_name. Data in the file should be stored neon model."""

    vec = []
    # Load a stored model file from disk (should have extension prm)
    params = pkl.load(open(file_name, 'r'))

    num_linear_layers = len(params["layer_params_states"])

    # Neon layers are LinearLayer_0, BiasLayer_1, LinearLayer_4, ...
    # A linear layer is followed by a bias layer. We need to concat their weights

    for layer_idx in range(0, num_linear_layers, 2):
        # Make sure our model has biases activated, otherwise add zeros here
        b = params["layer_params_states"][layer_idx + 1]['params']['W']
        w = params["layer_params_states"][layer_idx]['params']['W']

        layer_vector = np.ravel(np.hstack((b, w)))
        [vec.append(nv) for nv in layer_vector]
    return vec


def model2string(neon_model_path):
    """Serialize the model at the path to a string representation."""
    
    weights = extract_weights(neon_model_path)
    return struct.pack('f' * len(weights), *weights)

def convert(neon_model_path, output_filename):
    """Convert the serialization of a neon model into an iOS MLP model."""

    with open(output_filename, "wb") as f:
        #  You can use 'd' for double and < or > to force endinness
        model_as_string = model2string(neon_model_path)
        f.write(model_as_string)

def write_layers_to_file(layers, output_filename):
    """Write the layer configuration of a model to the output file."""

    with open(output_filename, 'w') as f:
        data = " ".join(str(e) for e in layers)
        f.write(data)


def activation_from_name(name):
    if name == "relu":
        activation = Rectlin()
    elif name == "logistic" or name == "sigmoid":
        activation = Logistic()
    elif name == "tanh":
        activation = Tanh()
    else:
        raise Exception("Bad activation name: " + name)
    return activation


def parsing_layer(layer_str):
    init_norm = Uniform(low=-0.1, high=0.1)
    bias_init = Constant(val=1.0)

    elements = layer_str.split(" ")
    i = 2
    layers = []
    while i < len(elements):
        number_nodes = elements[i]
        activation_func = activation_from_name(elements [i+1])
        layers.append(Affine(
            nout=int(number_nodes),
            init=init_norm,
            bias=bias_init,
            activation=activation_func))
        if i < len(elements) - 2:
            # don't add this Dropout at the last layer
            layers.append(Dropout(
                name="do_" + str(i),
                keep=0.9))
        i += 2

    return layers
