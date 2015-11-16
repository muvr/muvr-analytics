import sys
import argparse
import os
import zipfile
import csv
import tempfile

from training.acceleration_dataset import CSVAccelerationDataset
from training.examples import ExampleColl

mapping_exercise = {
    "arms_bicep-curl" : ["arms/bicep-curl", "bc", "bicep", "bicep curls", "biceps curls (left)"],
    "arms_tricep-extension": ["arms/tricep-extension", "tc", "te", "tricep"],
    "arms_lateral-raise": ["arms/lateral-raise", "lr", "lateral", "lateral raises"]
}

NO_EXERCISE_LABEL = "no_exercise"

def map_label(input):
    """Uniform the label"""
    if input == "":
        return NO_EXERCISE_LABEL
    else:
        format_input = input.strip().lower()
        output_name = input
        for standard, map_names in mapping_exercise.iteritems():
            if format_input in map_names:
                output_name = standard
        return output_name


def write_to_csv(filename, data):
    """Write csv data to filename"""
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def write_dataset_to_csv(dataset, output_directory):
    """Write dataset to multiple csv files"""
    csv = dataset.to_csv_data()
    for output, csv_data in csv:
        final_path = os.path.join(output_directory, output + ".csv")
        print "Writing", len(csv_data), "samples: ", final_path
        write_to_csv(final_path, csv_data)


def load_data(path, label_mapper):
    """ Load all csv files with converting label
        Return list of object ExampleColl, each has the same label
    """
    root_directory = ""
    if os.path.isdir(path):
        root_directory = path
    else:
        # Zip file - extract to temp root_directory first
        root_directory = tempfile.mkdtemp(prefix="/tmp/muvr-training-")
        zipfile.ZipFile(path, 'r').extractall(root_directory)

    csv_files = CSVAccelerationDataset.read_all_csv(root_directory)

    label_mapping = {}
    for f in csv_files:
        for label, x in CSVAccelerationDataset.load_example(f, label_mapper):
            data = label_mapping.get(label, [])
            data.append(x)
            label_mapping[label] = data

    result = []
    for label in label_mapping:
        xs = label_mapping[label]
        ys = [label] * len(xs)
        result.append(ExampleColl(xs, ys))
    return result


def main(dataset_directory, output_directory, ratio, is_slacking):
    """Main entry point."""

    # 1/ Load the dataset
    dataset = load_data(dataset_directory, map_label)
    training = []
    test = []
    for label_data in dataset:
        if is_slacking and label_data.labels[0] != NO_EXERCISE_LABEL:
            # training for slacking model, reset all label (bicep, tricep, ..) to exercise
            label_data.reset_all_labels("exercise")
        elif not is_slacking and label_data.labels[0] == NO_EXERCISE_LABEL:
            # training for exercise model, remove dataset with non-exercise label
            continue

        first, second = label_data.split(ratio/100.0)
        training.append(first)
        test.append(second)

    train_dataset = ExampleColl.concat(training)
    test_dataset = ExampleColl.concat(test)
    train_dataset.print_statistic("training dataset")
    test_dataset.print_statistic("testing dataset")

    write_dataset_to_csv(train_dataset, os.path.join(output_directory, "train"))
    write_dataset_to_csv(test_dataset, os.path.join(output_directory, "test"))



if __name__ == '__main__':
    """List arguments for this program"""
    parser = argparse.ArgumentParser(description='Preprocess the exercise dataset.')
    parser.add_argument('-d', metavar='dataset', type=str, help="folder containing exercise dataset")
    parser.add_argument('-o', metavar='output', default='./output', type=str, help="folder containing generated model")
    parser.add_argument('-ratio', metavar='ratio', default=80, type=int, help="train ratio")
    parser.add_argument('-slacking', action='store_true', default=False)
    args = parser.parse_args()

    sys.exit(main(args.d, args.o, args.ratio, args.slacking))
