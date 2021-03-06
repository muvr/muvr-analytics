"""Create new examples from existing one reproducing natural variation in the data"""

import numpy as np
from muvr.dataset.examples import ExampleColl
import logging


class SignalAugmenter(object):
    logger = logging.getLogger("training.SignalAugmenter")
    
    """Dataset augmentation specialized for one dimensional data.

    A sliding window will be used to move it over the examples to augment the dataset with new examples."""

    def __init__(self, augmentation_start, augmentation_end):
        self.augmentation_start = augmentation_start
        self.augmentation_end = augmentation_end

    def augment_examples(self, examples, target_feature_length):
        """Augments all the passed examples (should be a np array with 3 dimensions)."""
        augmented = []
        augmented_labels = []
        for i, features in enumerate(examples.features):
            if np.shape(features)[1] >= target_feature_length:
                label = examples.labels[i]
                one_augmented = self.augment_example(features, target_feature_length)
                factor = 1 if label == 0 else 1
                augmented.extend([one_augmented] * factor)
                augmented_labels.extend([label] * one_augmented.shape[0] * factor)
            else:
                self.logger.warn("Dropped an example because it was to short. Length: %d Expected: %d" % 
                                 (np.shape(features)[1], target_feature_length))

        if len(augmented) > 0:
            return ExampleColl(np.vstack(augmented), np.hstack(augmented_labels))
        else:
            return ExampleColl(np.empty((0, target_feature_length)), np.empty((0, 1)))
        
    def augment_example(self, example, target_length, window_step_size=5):
        """Example should be a numpy array, label a single label id."""
        dimensions = np.shape(example)[0]
        sample_length = np.shape(example)[1]

        # Those will define start and end of the data augmentation window, e.g. how far the window is moved over the
        # data. This assumes the first fraction of `augmentation_start` and the last fraction of `augmentation_end`
        # measurement points to be noise
        min_idx = int(sample_length * self.augmentation_start)
        max_idx = int(sample_length * self.augmentation_end)

        # Make sure we got enough data to get a complete example
        if (max_idx - min_idx) < target_length:
            start = (sample_length - target_length) / 2
            end = start + target_length
            return example[np.newaxis, :, start:end]
        else:
            augmentation_range = range(min_idx, max_idx - target_length, window_step_size)
            augmented = np.empty((len(augmentation_range), dimensions, target_length))
            # Use a sliding window to move it over the input example. this will create new examples that can be used for
            # training a model
            for i in augmentation_range:
                idx = (i - min_idx) / window_step_size
                augmented[idx, :, :] = example[:, i:(i + target_length)]
            return augmented
