import sys
from pyspark import SparkContext, SparkConf
from pyspark_cassandra import CassandraSparkContext
import os
from converters import neon2iosmlp
from training.acceleration_dataset import SparkAccelerationDataset
from training.mlp_model import MLPMeasurementModel
from itertools import groupby
from operator import attrgetter


def learn_model_from_data(dataset, working_directory, user_id):
    mlpmodel = MLPMeasurementModel(working_directory)

    trainedModel = mlpmodel.train(dataset)

    dataset.save_labels(os.path.join(working_directory, user_id + '_model.labels.txt'))
    neon2iosmlp.convert(mlpmodel.model_path, os.path.join(working_directory, user_id + '_model.weights.raw'))

    layers = mlpmodel.getLayer(dataset, trainedModel)
    neon2iosmlp.write_model_to_file(layers, os.path.join(working_directory, user_id + '_model.layers.txt'))

    return mlpmodel.model_path

def print_it(x):
    print x

def is_spark_dump_file(filename):
    return filename.startswith('part')

def trainModelOnUserData((user, rows)):
    user_id = user["user_id"]
    sorted_samples = sorted(rows, key=attrgetter("record_time"))
    train_examples = [list(samples) for _, samples in groupby(sorted_samples, key=attrgetter("record_id"))]

    dataset = SparkAccelerationDataset(train_examples, test_list=[])
    
    model = learn_model_from_data(dataset, os.path.join(os.path.abspath("../output"), 'models', user_id),user_id)
    
    return user, model

def main(sc):
    """Main entry point. Connects to cassandra and starts training."""
    lines = sc \
        .cassandraTable("muvrtest", "samples") \
        .select("user_id", "record_id", "record_time", "x", "y", "z", "label") \
        .spanBy('user_id') \
        .map(trainModelOnUserData) \
    # TODO: Decide what to do with the trained model
    

if __name__ == '__main__':
    # TODO: Where to get cluster ip from?
    # TODO: Which keyspace?
    conf = {
        'target_length': 400,
        'number_of_labels': 3,
        'cassandra_address': "localhost"
    }

    conf = SparkConf() \
        .setAppName('Muvr python spark training') \
        .setMaster('local[*]') \
        .set("spark.cassandra.connection.host", conf["cassandra_address"])

    # An external script needs to make sure that all the dependencies are packaged and provided to the workers!
    sc = CassandraSparkContext(conf=conf)

    sys.exit(main(sc))
