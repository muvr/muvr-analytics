package io.muvr.em.model

import java.util.UUID

import io.muvr.em.{ModelPreprocessing, ModelTemplate}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions

object MLP {

  def shallowModel: ModelTemplate = ModelTemplate("smlp@" + UUID.randomUUID().toString, newShallowModel, ModelPreprocessing.None)

  private def newShallowModel(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val distribution: UniformDistribution = new UniformDistribution(-0.1, 0.1)
    val conf = new NeuralNetConfiguration.Builder()
      .iterations(12)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .constrainGradientToUnitNorm(true)
      .learningRate(0.0001)
      .momentum(0.4)
      //.l1(0.1).l2(1)
      .regularization(true)
      .stepFunction(new NegativeDefaultStepFunction())
      .maxNumLineSearchIterations(10)
      .list(4)
      .layer(0, new DenseLayer.Builder()
        .nIn(numInputs)
        .nOut(900)
        .biasInit(1.0)
        .dropOut(0.9)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(900)
        .nOut(300)
        .dropOut(0.9)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .biasInit(1.0)
        .dist(distribution)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(300)
        .nOut(120)
        .dropOut(0.9)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .biasInit(1.0)
        .dist(distribution)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
        .nIn(120)
        .nOut(numOutputs)
        .biasInit(1.0)
        .dropOut(0.9)
        .activation("softmax")
        .weightInit(WeightInit.XAVIER)
        .dist(distribution)
        .build())
      .backprop(true).pretrain(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
