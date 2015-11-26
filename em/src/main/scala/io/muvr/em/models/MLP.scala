package io.muvr.em.models

import io.muvr.em.Model
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{Updater, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.stepfunctions.{NegativeDefaultStepFunction, NegativeGradientStepFunction, DefaultStepFunction, StepFunction}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions

object MLP {

  def shallowModel: Model = Model("smlp@" + System.currentTimeMillis(), newShallowModel)

  private def newShallowModel(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val distribution: UniformDistribution = new UniformDistribution(-0.1, 0.1)
    val conf = new NeuralNetConfiguration.Builder()
      .iterations(10)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .constrainGradientToUnitNorm(true)
      .learningRate(0.005)
      .momentum(0.4)
      .maxNumLineSearchIterations(10)
      .list(4)
      .layer(0, new DenseLayer.Builder()
        .nIn(numInputs)
        .nOut(500)
        .biasInit(1.0)
        .dropOut(0.9)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(500)
        .nOut(250)
        .dropOut(0.9)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .biasInit(1.0)
        .dist(distribution)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(250)
        .nOut(100)
        .dropOut(0.9)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .biasInit(1.0)
        .dist(distribution)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
        .nIn(100)
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
