package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object Model {
  type Id = String
  type Constructor = (Int, Int) => MultiLayerNetwork
}
/**
  * Model details
  * @param id identity
  * @param modelConstructor model constructor
  */
case class Model(id: Model.Id, modelConstructor: Model.Constructor)
