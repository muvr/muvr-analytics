from neon.backends import gen_backend
from neon.layers import Linear
import time
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Misclassification
import os
import logging
from training import utils
import yaml
from neon.util.yaml_parse import create_objects

class MLPMeasurementModel(object):
    """Wrapper around a neon MLP model that controls training parameters and configuration of the model."""

    random_seed = 666  # Take your lucky number
    
    Default_Batch_Size = 30
    Default_Max_Epochs = 10

    # Storage settings for the different output files
    Model_Filename = 'workout-mlp.pkl'
    Callback_Store_Filename = 'workout-mlp.h5'
    Intermediate_Model_Filename = 'workout-mlp-ep'

    def __init__(self, root_path, model_yaml_path):
        """Initialize paths and loggers of the model."""
        # Storage director of the model and its snapshots
        self.root_path = root_path
        self.model_path = os.path.join(self.root_path, self.Model_Filename)
        utils.remove_if_exists(self.model_path)

        self.model_yaml_path = model_yaml_path
        batch_size, epochs = self._load_backend_parameters_from(model_yaml_path)
        self.batch_size = batch_size
        self.max_epochs = epochs

        # Set logging output...
        for name in ["neon.util.persist"]:
            dslogger = logging.getLogger(name)
            dslogger.setLevel(40)
            
    def _load_backend_parameters_from(self, path):
        yaml_file = open(self.model_yaml_path, 'r')
        yaml_str = yaml_file.read()
        root_yaml = yaml.safe_load(yaml_str)
    
        epochs = root_yaml['epochs'] if 'epochs' in root_yaml else self.Default_Max_Epochs
        batch_size = root_yaml['batchsize'] if 'batchsize' in root_yaml else self.Default_Batch_Size
        return batch_size, epochs
    
    def _configure_callbacks(self, model, train, test):
        # Wrapper class to allow dynamic property changes
        class NeonCallbackParameters(object):
            pass
        
        args = NeonCallbackParameters()
        args.output_file = os.path.join(self.root_path, self.Callback_Store_Filename)
        args.evaluation_freq = 1
        args.progress_bar = True
        args.epochs = self.max_epochs
        args.save_path = os.path.join(self.root_path, self.Intermediate_Model_Filename)
        args.serialize = 1
        args.history = 100
        args.model_file = None
    
        callbacks = Callbacks(model, train, args, eval_set=test)
    
        # add a callback that saves the best model state
        callbacks.add_save_best_state_callback(self.model_path)
        return callbacks

    def train(self, dataset):
        """Trains the passed model on the given dataset. If no model is passed, `generate_default_model` is used."""
        print "Starting training..."
        start = time.time()

        # The training will be run on the CPU. If a GPU is available it should be used instead.
        backend = gen_backend(backend='cpu',
                              batch_size=30,
                              rng_seed=self.random_seed,
                              stochastic_round=False)

        # set up the model and experiment
        backend.bsz = self.batch_size
        model, cost, optimizer = create_objects(self.model_yaml_path)
        callbacks = self._configure_callbacks(model, dataset.train(), dataset.test())

        print 'Epochs: %d Batch-Size: %d' % (self.max_epochs, self.batch_size)
        model.fit(
            dataset.train(),
            optimizer=optimizer,
            num_epochs=self.max_epochs,
            cost=cost,
            callbacks=callbacks)

        print('Misclassification error = %.1f%%'
              % (model.eval(dataset.test(), metric=Misclassification()) * 100))
        print "Finished training!"
        end = time.time()
        print "Duration", end - start, "seconds"

        return model

    def getLayer(self, dataset, model):
        layerconfig = [dataset.num_features]
        for layer in model.layers.layers:
            if isinstance(layer, Linear):
                layerconfig.append(layer.nout)
        return layerconfig
