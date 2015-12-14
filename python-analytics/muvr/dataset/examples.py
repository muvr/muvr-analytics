import random
import numpy as np
from collections import Counter
import math
from muvr.util import utils


class ExampleColl(object):
    """Collection of training examples. Provides useful helpers to modify the collection."""

    # To make debugging easier, lets avoid randomness
    Seed = 42  # time()
    MAX_NUMBER_OF_EXAMPLE_REPETITIONS = 10

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

    def highpass_filter(self, rate, freq):
        """Apply a highpass filter to the data using the given parameters."""
        self.features = [np.apply_along_axis(utils.highpass_filter, 1, sample, rate, freq) for sample in self.features]
        
    def supersample_classes(self):
        """Supersample the classes to ensure a similar class distribution. 
        
        No example will be oversampled more than `MAX_NUMBER_OF_EXAMPLE_REPETITIONS` times. Be aware that, although
         aiming at a uniform distribution this will probably never achieve equal class sizes for all classes."""

        def sampling_factor(label_count, most_common):
            # Calculate the sampling factor. Factor will be always >= 0 and <= MAX_NUMBER_OF_EXAMPLE_REPETITIONS
            return max(min(int(math.ceil(most_common * 1.0 / label_count)), self.MAX_NUMBER_OF_EXAMPLE_REPETITIONS), 1)

        if len(self.labels) == 0:
            return

        label_dist = Counter(self.labels)
        most_common = label_dist.most_common(1)[0][1]  # Since we always have at least one label, this is save
            
        supersampled_features = []
        supersampled_labels = []
        for label, label_count in label_dist.iteritems():
            supersampling_factor = sampling_factor(label_count, most_common)
            # Collect all examples with the current label
            examples_with_label = [i for i, x in enumerate(self.labels) if x == label]
            
            for idx in examples_with_label:
                supersampled_features.extend([self.features[idx]] * supersampling_factor)
            supersampled_labels.extend([label] * supersampling_factor * len(examples_with_label))
            
        self.features = supersampled_features
        self.labels = supersampled_labels

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
