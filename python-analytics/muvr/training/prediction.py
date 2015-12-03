import numpy as np


def predict(model, dataset, num_labels):
    """Use the passed model to create a prediction for each example in the dataset."""
    
    dataset.reset()
    predictions = np.empty((num_labels, 0), dtype="float32")
    n_processed = 0
    for x, t in dataset:
        prediction = model.fprop(x, inference=True).asnumpyarray()
        bsz = min(dataset.ndata - n_processed, model.be.bsz)
        n_processed += bsz
        predictions = np.hstack((predictions, prediction[:, :bsz]))
    return predictions
