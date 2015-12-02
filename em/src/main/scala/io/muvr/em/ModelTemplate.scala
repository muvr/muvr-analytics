package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

/**
  * Model template types
  */
object ModelTemplate {
  type Constructor = (Int, Int) ⇒ MultiLayerNetwork
}

/**
  * Options for model preprocessing
  */
sealed trait ModelPreprocessing
object ModelPreprocessing {
  case object None extends ModelPreprocessing
}

/**
  * Model details
  * @param id identity
  * @param modelConstructor model constructor
  * @param preprocessingSteps the pre-processing to be applied
  */
case class ModelTemplate(id: ModelId, modelConstructor: ModelTemplate.Constructor, preprocessingSteps: List[ModelPreprocessing] = Nil)
