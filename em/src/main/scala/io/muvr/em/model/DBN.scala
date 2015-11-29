package io.muvr.em.model

import java.util.Collections

import io.muvr.em.{ModelPreprocessing, ModelTemplate}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object DBN {

  val model: ModelTemplate = ModelTemplate("dbn", newModel, ModelPreprocessing.None)

  private def newModel(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val iterations = 10

    val conf = new NeuralNetConfiguration.Builder()
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .iterations(iterations)
      .momentum(0.5)
      .momentumAfter(Collections.singletonMap(3, 0.9))
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .list(4)
      .layer(0, new RBM.Builder().nIn(numInputs).nOut(500)
        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
        .visibleUnit(RBM.VisibleUnit.BINARY)
        .hiddenUnit(RBM.HiddenUnit.BINARY)
        .build())
      .layer(1, new RBM.Builder().nIn(500).nOut(250)
        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
        .visibleUnit(RBM.VisibleUnit.BINARY)
        .hiddenUnit(RBM.HiddenUnit.BINARY)
        .build())
      .layer(2, new RBM.Builder().nIn(250).nOut(200)
        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
        .visibleUnit(RBM.VisibleUnit.BINARY)
        .hiddenUnit(RBM.HiddenUnit.BINARY)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
        .nIn(200).nOut(numOutputs).build())
      .pretrain(true).backprop(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    model
  }

}
