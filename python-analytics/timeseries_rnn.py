import numpy as np
import math
from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, RecurrentLast, Recurrent
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Logistic, Tanh, Identity, MeanSquared, Softmax
from neon.callbacks.callbacks import Callbacks
from neon import NervanaObject
from neon.util.argparser import NeonArgparser, extract_valid_args

def rolling_window(a, lag):
    """
    Convert a into time-lagged vectors

    a    : (n, p)
    lag  : time steps used for prediction

    returns  (n-lag+1, lag, p)  array

    (Building time-lagged vectors is not necessary for neon.)
    """
    assert a.shape[0] > lag

    shape = [a.shape[0] - lag + 1, lag, a.shape[-1]]
    strides = [a.strides[0], a.strides[0], a.strides[-1]]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class DataIteratorSequence(NervanaObject):

    """
    This class takes a sequence and returns an iterator providing data in batches suitable for RNN
    prediction.  Meant for use when the entire dataset is small enough to fit in memory.
    """

    def __init__(self, X, time_steps, forward=1, return_sequences=True):
        """
        Implements loading of given data into backend tensor objects. If the backend is specific
        to an accelerator device, the data is copied over to that device.

        Args:
            X (ndarray): Input sequence with feature size within the dataset.
                         Shape should be specified as (num examples, feature size]
            time_steps (int): The number of examples to be put into one sequence.
            forward (int, optional): how many forward steps the sequence should predict. default
                                     is 1, which is the next example
            return_sequences (boolean, optional): whether the target is a sequence or single step.
                                                  Also determines whether data will be formatted
                                                  as strides or rolling windows.
                                                  If true, target value be a sequence, input data
                                                  will be reshaped as strides.  If false, target
                                                  value will be a single step, input data will be
                                                  a rolling_window
        """
        self.seq_length = time_steps
        self.forward = forward
        self.batch_index = 0
        self.nfeatures = self.nclass = X.shape[1]
        self.nsamples = X.shape[0]
        self.shape = (self.nfeatures, time_steps)
        self.return_sequences = return_sequences

        target_steps = time_steps if return_sequences else 1
        # pre-allocate the device buffer to provide data for each minibatch
        # buffer size is nfeatures x (times * batch_size), which is handled by backend.iobuf()
        self.X_dev = self.be.iobuf((self.nfeatures, time_steps))
        self.y_dev = self.be.iobuf((self.nfeatures, target_steps))

        if return_sequences is True:
            # truncate to make the data fit into multiples of batches
            extra_examples = self.nsamples % (self.be.bsz * time_steps)
            if extra_examples:
                X = X[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples / (self.be.bsz * time_steps)
            self.ndata = self.nbatches * self.be.bsz * time_steps  # no leftovers

            # y is the lagged version of X
            y = np.concatenate((X[forward:], X[:forward]))
            self.y_series = y
            # reshape this way so sequence is continuous along the batches
            self.X = X.reshape(self.be.bsz, self.nbatches, time_steps, self.nfeatures)
            self.y = y.reshape(self.be.bsz, self.nbatches, time_steps, self.nfeatures)
        else:
            self.X = rolling_window(X, time_steps)
            self.X = self.X[:-1]
            self.y = X[time_steps:]

            self.nsamples = self.X.shape[0]
            extra_examples = self.nsamples % (self.be.bsz)
            if extra_examples:
                self.X = self.X[:-extra_examples]
                self.y = self.y[:-extra_examples]

            # calculate how many batches
            self.nsamples -= extra_examples
            self.nbatches = self.nsamples / (self.be.bsz)
            self.ndata = self.nbatches * self.be.bsz
            self.y_series = self.y

            Xshape = (self.nbatches, self.be.bsz, time_steps, self.nfeatures)
            Yshape = (self.nbatches, self.be.bsz, 1, self.nfeatures)
            self.X = self.X.reshape(Xshape).transpose(1, 0, 2, 3)
            self.y = self.y.reshape(Yshape).transpose(1, 0, 2, 3)



    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of data.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            # get the data for this batch and reshape to fit the device buffer shape
            X_batch = self.X[:, self.batch_index].reshape(self.X_dev.shape[::-1]).T.copy()
            y_batch = self.y[:, self.batch_index].reshape(self.y_dev.shape[::-1]).T.copy()

            # make the data for this batch as backend tensor
            self.X_dev.set(X_batch)
            self.y_dev.set(y_batch)

            self.batch_index += 1

            yield self.X_dev, self.y_dev


# replicate neon's mse error metric
def err(y, t):
    feature_axis = 1
    return (0.5 * np.square(y - t).mean(axis=feature_axis).mean())


# input file contains 3 lines which correspond to train, valid and test sequence
def parseData(filename):
	with open(filename, 'r') as myfile:
		data = myfile.readlines()
		train_sequence = np.array([[float(s)] for s in data[0].strip().split(" ")])
		valid_sequence = np.array([[float(s)] for s in data[1].strip().split(" ")])
        test_sequence = np.array([[float(s)] for s in data[2].strip().split(" ")])
	return (train_sequence, valid_sequence, test_sequence)

if __name__ == '__main__':

    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    # parser.add_argument('--curvetype', default='Lissajous1', choices=['Lissajous1', 'Lissajous2'],
    #                     help='type of input curve data to use (Lissajous1 or Lissajous2)')
    args = parser.parse_args(gen_be=False)

    # network hyperparameters
    hidden = 32
    args.batch_size = 1

    # The following flag will switch between 2 training strategies:
    # 1. return_sequence True:
    #       Inputs are sequences, and target outputs will be sequences.
    #       The RNN layer's output at EVERY step will be used for errors and optimized.
    #       The RNN model contains a RNN layer and an Affine layer
    #       The data iterator will format the data accordingly, and will stride along the
    #           whole series with no overlap
    # 2. return_sequence False:
    #       Inputs are sequences, and target output will be a single step.
    #       The RNN layer's output at LAST step will be used for errors and optimized.
    #       The RNN model contains a RNN layer and RNN-output layer (i.g. RecurrentLast, etc.)
    #           and an Affine layer
    #       The data iterator will format the data accordingly, using a rolling window to go
    #           through the data
    return_sequences = False

    # Note that when the time series has higher or lower frequency, it requires different amounts
    # of data to learn the temporal pattern, the sequence length and the batch size for the
    # training process also makes a difference on learning performance.

    seq_len = 5

    # ================= Main neon script ====================

    be = gen_backend(**extract_valid_args(args, gen_backend))

    # a file to save the trained model
    if args.save_path is None:
        args.save_path = 'timeseries.pkl'

    if args.callback_args['save_path'] is None:
        args.callback_args['save_path'] = args.save_path

    if args.callback_args['serialize'] is None:
        args.callback_args['serialize'] = 1


    seq_len = 3
    (train_data, valid_data, test_data) = parseData("input.txt")
    max_val = max(train_data.max(), valid_data.max(), test_data.max()) + 15
    min_val = min(train_data.min(), valid_data.min(), test_data.min()) - 5

    # normalize the input into range [0, 1]
    def normalize(input):
        def transform(x):
            return (x - min_val) / (max_val - min_val)
        return transform(input)

    # convert the normalize number to original range
    def convert(input):
		return input * (max_val - min_val) + min_val

    # normalize input
    train_data_n = normalize(train_data)
    valid_data_n = normalize(valid_data)

    # use data iterator to feed X, Y. return_sequence determines training strategy
    train_set = DataIteratorSequence(train_data_n, seq_len, return_sequences=return_sequences)
    valid_set = DataIteratorSequence(valid_data_n, seq_len, return_sequences=return_sequences)

    print "\nNumber of features:", train_set.nfeatures
    print "Next value depends on", seq_len, "past number"
    print "max_range =", max_val
    print "min_range =", min_val

    # define weights initialization
    init = GlorotUniform()  # Uniform(low=-0.08, high=0.08)

    # define model: model is different for the 2 strategies (sequence target or not)
    if return_sequences is True:
        layers = [
            LSTM(hidden, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=False),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]
    else:
        layers = [
            LSTM(hidden, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=True),
            RecurrentLast(),
            Affine(train_set.nfeatures, init, bias=init, activation=Identity())
        ]


    model = Model(layers=layers)
    cost = GeneralizedCost(MeanSquared())
    optimizer = RMSProp(stochastic_round=args.rounding, learning_rate=0.02, decay_rate=0.6)

    callbacks = Callbacks(model, train_set, eval_set=valid_set, **args.callback_args)

    # fit model
    print "\nSTART TRAINING\n"
    model.fit(train_set,
              optimizer=optimizer,
              num_epochs=3,
              cost=cost,
              callbacks=callbacks)

    # run the trained model on train and valid dataset and see how the outputs match
    train_output = model.get_outputs(train_set).reshape(-1, train_set.nfeatures)
    valid_output = model.get_outputs(valid_set).reshape(-1, valid_set.nfeatures)
    train_target = train_set.y_series
    valid_target = valid_set.y_series

    # calculate accuracy
    terr = err(train_output, train_target)
    verr = err(valid_output, valid_target)

    print '\ntrain err = %g, valid err = %g' % (terr, verr)

    print "\nNow test with sequence:", test_data.flatten()
    test_set = DataIteratorSequence(normalize(test_data), seq_len, return_sequences=return_sequences)

    overlap_sequence = rolling_window(test_data, seq_len)[:-1]
    test_output = model.get_outputs(test_set)
    convert_output = convert(test_output)
    for index, test_seq in enumerate(overlap_sequence):
        print "\nNext value of sequence:", test_seq.flatten()
        print convert_output[index]

