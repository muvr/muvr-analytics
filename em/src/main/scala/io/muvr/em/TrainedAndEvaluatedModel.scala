package io.muvr.em

import io.muvr.em.dataset.Labels
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

/**
  * Holds the trained model, its labels and the result of its evaluation
  *
  * @param id the model id
  * @param model the model
  * @param labels the labels
  * @param evaluation the evaluation
  */
case class TrainedAndEvaluatedModel(id: ModelId, model: MultiLayerNetwork, labels: Labels, evaluation: ModelEvaluation)
