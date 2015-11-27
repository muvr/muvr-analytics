package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object ModelTemplate {
  type Id = String
  type Constructor = (Int, Int) ⇒ MultiLayerNetwork
}

/**
  * Model details
  * @param id identity
  * @param modelConstructor model constructor
  */
case class ModelTemplate(id: ModelTemplate.Id, modelConstructor: ModelTemplate.Constructor)
