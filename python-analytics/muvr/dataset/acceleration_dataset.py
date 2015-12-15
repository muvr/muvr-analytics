import collections
import numpy as np
from neon.data import DataIterator
import logging
import os
import csv
import zipfile
import tempfile
from muvr.dataset.augmentation import SignalAugmenter
from muvr.dataset.examples import ExampleColl
from collections import defaultdict


class AccelerationDataset(object):
    """Dataset containing examples based on acceleration data and their labels."""
    logger = logging.getLogger("training.AccelerationDataset")

    # Defines the ratio of examples used for training. The rest will be reserved for testing.
    TRAIN_RATIO = 0.8

    # Target feature length of the examples during training
    Target_Feature_Length = 400

    # -- Signal Preprocessing Settings --
    # This defines the range of the values the accelerometer measures
    Feature_Range = 4.0
    Feature_Mean = 0
    # Rate at which the data got collected in seconds per sample
    Feature_Sample_Rate = 1.0/50
    # Defines the highest possible frequency, everything else will get filtered by the highpass filter
    Highpass_Filter_Cutoff = 1.0/10

    def human_label_for(self, label_id):
        """Convert a label id into a human readable string label."""
        return self.id_label_mapping[label_id]

    def ordered_labels(self):
        return collections.OrderedDict(sorted(self.id_label_mapping.items())).values()

    def save_labels(self, filename):
        """Store the label <--> id mapping to file. The id is defined by the line number."""
        labels = self.ordered_labels()

        with open(filename, 'wb') as f:
            f.write("\n".join(labels))

    def prepare_dataset(self, dataset, supersample=False):
        """Prepare the loaded data for the neural network training.
        
        This is a rather important step and care should be taken to ensure the same preprocessing steps are applied
        when the trained model is used. This includes signal filtering, and scaling of the signal.
        """
        self.logger.info("Loading DS from files...")
        
        dataset.highpass_filter(rate=self.Feature_Sample_Rate, freq=self.Highpass_Filter_Cutoff)
        
        if supersample:
            dataset.supersample_classes()

        augmented = self.augmenter.augment_examples(dataset, self.Target_Feature_Length)
        self.logger.info("Augmented with %d examples, %d originally" % (
            augmented.num_examples - dataset.num_examples, dataset.num_examples))

        if augmented.num_examples > 0:
            augmented.shuffle()
            augmented.scale_features(self.Feature_Range, self.Feature_Mean)

        return augmented

    # Load label mapping and train / test data from disk.
    def __init__(self, train_examples, test_examples=None, supersample_in_train=True):
        """Initialize the dataset using the provided train and test examples."""

        self.logger.info("Loading DS from files...")
        self.augmenter = SignalAugmenter(augmentation_start=0.1, augmentation_end=0.9)

        train = self.prepare_dataset(train_examples, supersample=supersample_in_train)
        test = self.prepare_dataset(test_examples, supersample=False)

        self.id_label_mapping = {v: k for k, v in self.label_id_mapping.items()}
        self.X_train = self.flatten2d(train.features)
        self.y_train = train.labels
        self.X_test = self.flatten2d(test.features)
        self.y_test = test.labels

        self.num_labels = len(self.id_label_mapping)
        self.num_features = self.X_train.shape[1]
        self.num_train_examples = self.X_train.shape[0]
        self.num_test_examples = self.X_test.shape[0]

        self.train_examples = train_examples
        self.test_examples = test_examples

    @staticmethod
    def flatten2d(npa):
        """Take a 3D array and flatten the last dimension."""
        if npa.shape[0] > 0:
            return npa.reshape((npa.shape[0], -1))
        else:
            return npa

    # Get the dataset ready for Neon training
    def train(self):
        """Provide neon data iterator for training purposes."""
        return DataIterator(
            X=self.X_train,
            y=self.y_train,
            nclass=self.num_labels,
            make_onehot=True,
            lshape=(3, self.num_features / 3, 1))

    def test(self):
        """Provide neon data iterator for testing purposes."""
        if self.num_test_examples > 0:
            return DataIterator(
                X=self.X_test,
                y=self.y_test,
                nclass=self.num_labels,
                make_onehot=True,
                lshape=(3, self.num_features / 3, 1))
        else:
            return None


class SparkAccelerationDataset(AccelerationDataset):
    def __init__(self, example_list, label_mapper=lambda x: x, supersample_in_train=True):
        """Load the data from the provided nested list of examples."""
        self.label_id_mapping = {}
        examples = self.transform_to_example_coll(example_list, label_mapper)
        examples.shuffle()

        train, test = examples.split(self.TRAIN_RATIO)

        super(SparkAccelerationDataset, self).__init__(train, test, supersample_in_train)

    def transform_to_example_coll(self, examples, label_mapper):
        def transform(example):
            """Load a single example from a CSV file."""
            single_examples = []
            x_buffer = []
            last_label = None
            for row in example:
                label = row["exercise"]
                if last_label and label != last_label:
                    x = np.transpose(np.reshape(np.asarray(x_buffer, dtype=float), (len(x_buffer), len(x_buffer[0]))))
                    if label_mapper(last_label):
                        single_examples.append((label_mapper(last_label), x))
                    x_buffer = []
                x_buffer.append([row["x"], row["y"], row["z"]])
                last_label = label

            if len(x_buffer) > 0:
                x = np.transpose(np.reshape(np.asarray(x_buffer, dtype=float), (len(x_buffer), len(x_buffer[0]))))
                if label_mapper(last_label):
                    single_examples.append((label_mapper(last_label), x))

            return single_examples

        xs = []
        ys = []
        for example in examples:
            for label, x in transform(example):
                if label not in self.label_id_mapping:
                    self.label_id_mapping[label] = len(self.label_id_mapping)
                xs.append(x)
                ys.append(self.label_id_mapping[label])

        return ExampleColl(xs, ys)

class CSVAccelerationDataset(AccelerationDataset):
    def __init__(self, directory, test_directory=None, label_mapper=lambda x: x, supersample_in_train=True):
        """Load the dataset data from the directory.

        If two directories are passed the second is interpreted as the test dataset. If only one dataset gets passed,
         this dataset will get split into test and train. The label_mapper`allows to modify loaded labels. This is
         useful e.g. to map multiple labels to a single on ("arms/biceps-curl" --> "-/exercising", ...)."""

        # If we get provided with a test directory, we are going to use that. Otherwise we will split the dataset in
        # test and train on our own.
        self.label_id_mapping = {}
        if test_directory:
            train = ExampleColl.concat(self.load_examples(directory, label_mapper))
            test = ExampleColl.concat(self.load_examples(test_directory, label_mapper))
        else:
            examples = self.load_examples(directory, label_mapper)
            # examples.shuffle()

            training_data = []
            test_data = []
            for label_data in examples:
                first, second = label_data.split(self.TRAIN_RATIO)
                training_data.append(first)
                test_data.append(second)

            train = ExampleColl.concat(training_data).shuffle()
            test = ExampleColl.concat(test_data).shuffle()

        super(CSVAccelerationDataset, self).__init__(train, test, supersample_in_train)

    def load_examples(self, path, label_mapper):
        """
        Load examples contained in the path into an example collection. Examples need to be stored in CSVs.

        The label_mapper allows to map a loaded label to a different label, e.g. to combine multiple labels into one.

        Arguments:

        :param path: can be directory or zipfile. If zipfile, it will be extracted to a temporary path with prefix ``/tmp/muvr-training-``
        :param label_mapper: the mapper
        :return: list of object ExampleColl, each has the same label
        """
        root_directory = ""
        if os.path.isdir(path):
            root_directory = path
        else:
            # Zip file - extract to temp root_directory first
            root_directory = tempfile.mkdtemp(prefix="/tmp/muvr-training-")
            zipfile.ZipFile(path, 'r').extractall(root_directory)

        csv_files = []

        for root, _, files in os.walk(root_directory, followlinks=True):
            for name in files:
                f = os.path.join(root, name)
                if os.path.isfile(f) and f.endswith("csv"):
                    csv_files.append(f)

        label_mapping = {}
        for f in csv_files:
            for label, x in self.load_example(f, label_mapper):
                if label not in self.label_id_mapping:
                    self.label_id_mapping[label] = len(self.label_id_mapping)
                label_id = self.label_id_mapping[label]
                samples = label_mapping.get(label_id, [])
                samples.append(x)
                label_mapping[label_id] = samples

        result = []
        for label in label_mapping:
            xs = label_mapping[label]
            ys = [label] * len(xs)
            result.append(ExampleColl(xs, ys))
        return result

    @staticmethod
    def load_example(filename, label_mapper):
        """Load a single example from a CSV file.
        Return a dictionary object with key is label, value is the list of data with that label"""
        single_examples = []

        with open(filename, 'rb') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            csv_data = csv.reader(csvfile, dialect)

            x_buffer = []
            last_label = None
            for row in csv_data:
                if len(row) == 7:
                    # New format (len = 7)
                    #   X | Y | Z | '' | '' | '' | ''(without label)
                    #   X | Y | Z | biceps-curl | intensity | weight | repetition
                    new_data = row[0:3]
                    label = row[3]
                    if last_label is not None and label != last_label:
                        x = np.transpose(np.reshape(np.asarray(x_buffer, dtype=float), (len(x_buffer), len(x_buffer[0]))))
                        if label_mapper(last_label):
                            single_examples.append((label_mapper(last_label), x))
                        x_buffer = []
                    x_buffer.append(new_data)
                    last_label = label
                else:
                    raise Exception("Bad format with file: " + filename)

            if len(x_buffer) > 0:
                x = np.transpose(np.reshape(np.asarray(x_buffer, dtype=float), (len(x_buffer), len(x_buffer[0]))))
                if label_mapper(last_label):
                    single_examples.append((label_mapper(last_label), x))

        return single_examples
