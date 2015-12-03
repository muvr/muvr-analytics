from neon.initializers import Uniform, Constant
from neon.layers import Affine, Dropout
from neon.models import Model
from neon.transforms import Rectlin, Logistic


def generate_default_exercise_model(num_labels):
    """Generate layers and a MLP model using the given settings."""
    init_norm = Uniform(low=-0.1, high=0.1)
    bias_init = Constant(val=1.0)

    layers = []
    layers.append(Affine(
        nout=250,
        init=init_norm,
        bias=bias_init,
        activation=Rectlin()))

    layers.append(Dropout(
        name="do_1",
        keep=0.9))

    layers.append(Affine(
        nout=50,
        init=init_norm,
        bias=bias_init,
        activation=Rectlin()))

    layers.append(Dropout(
        name="do_3",
        keep=0.9))

    layers.append(Affine(
        nout=num_labels,
        init=init_norm,
        bias=bias_init,
        activation=Logistic()))

    return Model(layers=layers)


def generate_default_activity_model(num_labels):
    """Generate layers and a MLP model using the given settings."""
    init_norm = Uniform(low=-0.1, high=0.1)
    bias_init = Constant(val=1.0)

    layers = []
    layers.append(Affine(
        nout=250,
        init=init_norm,
        bias=bias_init,
        activation=Rectlin()))

    layers.append(Dropout(
        name="do_1",
        keep=0.95))

    layers.append(Affine(
        nout=50,
        init=init_norm,
        bias=bias_init,
        activation=Rectlin()))

    layers.append(Dropout(
        name="do_2",
        keep=0.95))

    layers.append(Affine(
        nout=num_labels,
        init=init_norm,
        bias=bias_init,
        activation=Logistic()))

    return Model(layers=layers)
