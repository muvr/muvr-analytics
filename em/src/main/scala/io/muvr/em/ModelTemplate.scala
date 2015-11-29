package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object ModelTemplate {
  type Id = String
  type Constructor = (Int, Int) ⇒ MultiLayerNetwork
}

sealed trait ModelPreprocessing
object ModelPreprocessing {
  case object None extends ModelPreprocessing
}

/**
  * Model details
  * @param id identity
  * @param modelConstructor model constructor
  * @param preprocessing the pre-processing to be applied
  */
case class ModelTemplate(id: ModelTemplate.Id, modelConstructor: ModelTemplate.Constructor, preprocessing: ModelPreprocessing)
