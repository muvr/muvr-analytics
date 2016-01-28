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


# input file contains multiple lines, each line is the sequence of number in one session
def parseData(filename):
	with open(filename, 'r') as myfile:
		data = myfile.readlines()
        def parseLine(line, separator):
            return np.array([[float(s)] for s in line.strip().split(separator)])
        sequences = [parseLine(line, ",") for line in data]
	return sequences


def get_max(list_data):
    max_number = float("-inf")
    for data in list_data:
        if data.max() > max_number:
            max_number = data.max()
    return max_number


def get_min(list_data):
    min_number = float("inf")
    for data in list_data:
        if data.min() < min_number:
            min_number = data.min()
    return min_number


# normalize the input into range [0, 1]
def normalize(input, min_val, max_val):
    def transform(x):
        return (x - min_val) / (max_val - min_val)
    return transform(input)


# convert the normalize number to original range
def convert(input, min_val, max_val):
    return input * (max_val - min_val) + min_val


def duplicate_data(array_input, limit):
    number_duplicate = limit / len(array_input)
    new_array = array_input
    for i in range(2, number_duplicate+1):
        new_array = np.concatenate((new_array, array_input))
    return new_array


def build_dataset(array_input, seq_len, min_val, max_val, return_sequences):
    normalize_array = normalize(array_input, min_val, max_val)
    dataset = DataIteratorSequence(normalize_array, seq_len, return_sequences=return_sequences)
    return dataset


def setup_model(num_features, hidden, return_sequences):
    # define weights initialization
    init = GlorotUniform()  # Uniform(low=-0.08, high=0.08)

    # define model: model is different for the 2 strategies (sequence target or not)
    if return_sequences is True:
        layers = [
            LSTM(hidden, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=False),
            Affine(num_features, init, bias=init, activation=Identity())
        ]
    else:
        layers = [
            LSTM(hidden, init, activation=Logistic(), gate_activation=Tanh(), reset_cells=True),
            RecurrentLast(),
            Affine(num_features, init, bias=init, activation=Identity())
        ]

    model = Model(layers=layers)
    return model


def run_training(model, dataset, valid_set, args, cost, optimizer):
    callbacks = Callbacks(model, dataset, eval_set=valid_set, **args.callback_args)
    model.fit(dataset, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)


def predict_next_value(model, test_data, seq_len, min_val, max_val, return_sequences):
    lst = test_data.tolist()
    lst.append([0])
    test_data = np.array(lst)

    test_set = build_dataset(test_data, seq_len, min_val, max_val, return_sequences)

    overlap_sequence = rolling_window(test_data, seq_len)[:-1]
    test_output = model.get_outputs(test_set)
    convert_output = convert(test_output, min_val, max_val)
    return convert_output[len(convert_output) - 1][0]

def roundMinMax(value, minimum, step, maximum):
    if value < minimum:
        return minimum

    if value >= maximum:
        return maximum

    weight = minimum
    while weight < maximum:
        dcw = value - weight
        dnw = value - (weight + step)
        if dcw >= 0 and dnw <= 0:
            # value is in range
            if abs(dcw) > abs(dnw):
                return weight + step
            else:
                return weight

        weight += step

    return value


def error_by_last_value(data):
    return sum((data[0:(len(data)-1)] - data[1:len(data)]) ** 2)


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
    limit_size = 200
    list_data = parseData("input.txt")

    max_val = get_max(list_data) + 15
    min_val = get_min(list_data) - 5

    valid_data = list_data[0]
    valid_set = build_dataset(valid_data, seq_len, min_val, max_val, return_sequences)

    model = setup_model(valid_set.nfeatures, hidden, return_sequences)
    args.epochs = 0

    cost = GeneralizedCost(MeanSquared())
    optimizer = RMSProp(stochastic_round=args.rounding, learning_rate=0.02, decay_rate=0.6)

    list_error = []
    for data in list_data:
        # training for each session
        print "START TRAINING for this session:", data.flatten(), "length =", len(data)
        #error_rate = error_by_last_value(data[0:seq_len+1])
        error_rate = 0
        predicted_sequence = np.array([[0]])
        predicted_sequence = np.concatenate((predicted_sequence, data[0:seq_len]))
        for index in range(seq_len, len(data)):
            # training incrementally in one session
            train_data = data[0:(index+1)]
            print "Train incrementally for this subset:", train_data.flatten()
            new_train = duplicate_data(train_data, limit_size)
            train_set = build_dataset(new_train, seq_len, min_val, max_val, return_sequences)
            
            args.epochs += 3
            run_training(model, train_set, valid_set, args, cost, optimizer)

            predict_val = predict_next_value(model, train_data, seq_len, min_val, max_val, return_sequences)
            predict_val = roundMinMax(predict_val, 2.5, 2.5, 999)
            if index < len(data) - 1:
                print "==>", predict_val, "vs expect", data[index+1]
                error_rate += (predict_val - data[index+1]) ** 2
                predicted_sequence = np.concatenate((predicted_sequence, np.array([[predict_val]])))
            print
        
        list_error.append(error_rate)
        print "input_sequence =", data.flatten()
        print "predicted_sequence =", predicted_sequence.flatten(), "\n"

    for error in list_error:
        print "Error rate:", error

    

    # print "\nNumber of features:", train_set.nfeatures
    # print "Next value depends on", seq_len, "past number"
    # print "max_range =", max_val
    # print "min_range =", min_val

    # # run the trained model on train and valid dataset and see how the outputs match
    # train_output = model.get_outputs(train_set).reshape(-1, train_set.nfeatures)
    # valid_output = model.get_outputs(valid_set).reshape(-1, valid_set.nfeatures)
    # train_target = train_set.y_series
    # valid_target = valid_set.y_series

    # # calculate accuracy
    # terr = err(train_output, train_target)
    # verr = err(valid_output, valid_target)

    # print '\ntrain err = %g, valid err = %g' % (terr, verr)


