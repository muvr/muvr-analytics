import random
import numpy as np
import collections
import itertools
import logging

class ExampleColl(object):
    """Collection of training examples. Provides useful helpers to modify the collection."""

    logger = logging.getLogger("training.ExampleColl")
    logger.setLevel(logging.INFO)

    # To make debugging easier, lets avoid randomness
    Seed = 42  # time()

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_examples = len(features)
        random.seed(self.Seed)

    def split(self, ratio):
        """Split the collection into two parts.

        The first part will be of the given ratio. Second part will contain the remaining examples."""
        split_point = int(self.num_examples * ratio)
        first = ExampleColl(self.features[0:split_point], self.labels[0:split_point])
        second = ExampleColl(self.features[split_point:self.num_examples], self.labels[split_point:self.num_examples])
        return first, second

    def scale_features(self, feature_range, feature_mean):
        """Scale the features of the examples using the passed range and mean."""
        self.features = np.divide(np.subtract(self.features, feature_mean), feature_range / 2.0)

    def shuffle(self):
        """Shuffle the examples in this collection randomly."""
        shuffled_idx = range(self.num_examples)
        random.shuffle(shuffled_idx)

        shuffled = []
        shuffled_labels = []
        for i in shuffled_idx:
            shuffled.append(self.features[i])
            shuffled_labels.append(self.labels[i])

        if isinstance(self.features, list):
            self.features = shuffled
            self.labels = shuffled_labels
        else:
            self.features = np.reshape(np.array(shuffled), (len(shuffled_labels),) + shuffled[0].shape)
            self.labels = np.reshape(np.array(shuffled_labels), (len(shuffled_labels)))

    def print_statistic(self, example_name = "example", label_id_mapping = None):
        self.logger.info("Statistic of %s: %d labelled data" % (example_name, self.num_examples))
        counter = collections.Counter(self.labels)

        if label_id_mapping is None:
            for label, count in counter.iteritems():
                self.logger.info("\t%s --> %d" % (label, count))
        else:
            for label, id in label_id_mapping.items():
                self.logger.info("\t%s --> %d" % (label, counter[id]))
        self.logger.info("\n")

    def reset_all_labels(self, new_labels):
        self.labels = [new_labels] * self.num_examples

    def to_csv_data(self, label_id_mapping = None):
        csv_data = []
        for idx, matrix in enumerate(self.features):
            if label_id_mapping is None:
                label = self.labels[idx]
            else:
                id_label_mapping = {v: k for k, v in label_id_mapping.items()}
                label = id_label_mapping[self.labels[idx]]
            output_filename = str(label) + "_" + str(idx)
            single_csv = []
            for one_column in matrix.T:
                # Format of csv X | Y | Z | biceps-curl | intensity | weight | repetition
                labelled_data = list(one_column)
                labelled_data.append(label)
                labelled_data.append("")
                labelled_data.append("")
                labelled_data.append("")
                single_csv.append(labelled_data)
            csv_data.append((output_filename, single_csv))
        return csv_data

    @staticmethod
    def concat(array_example):
        features = []
        labels = []
        for example in array_example:
            features = itertools.chain(features, example.features)
            labels = itertools.chain(labels, example.labels)
        return ExampleColl(list(features), list(labels))