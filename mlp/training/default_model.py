from neon.initializers import Uniform, Constant
from neon.layers import Affine, Dropout, GeneralizedCost, Linear
from neon.transforms import Rectlin, Logistic, Tanh
from neon.models import Model

class DefaultModel(object):

    @staticmethod
    def get_slacking_model(num_labels):
        """ Generate layers and a MLP model for exercise vs no-exercise"""
        init_norm = Uniform(low=-0.1, high=0.1)
        bias_init = Constant(val=1.0)

        layers = []

        layers.append(Affine(
            nout=500,
            init=init_norm,
            bias=bias_init,
            activation=Tanh()))

        layers.append(Dropout(
            name="do_1",
            keep = 0.9))

        layers.append(Affine(
            nout=100,
            init=init_norm,
            bias=bias_init,
            activation=Tanh()))

        layers.append(Dropout(
            name="do_2",
            keep = 0.9))

        layers.append(Affine(
            nout=25,
            init=init_norm,
            bias=bias_init,
            activation=Tanh()))

        layers.append(Dropout(
            name="do_3",
            keep = 0.9))

        layers.append(Affine(
            nout = num_labels,
            init=init_norm,
            bias=bias_init,
            activation=Logistic()))

        model = Model(layers=layers)
        return model
