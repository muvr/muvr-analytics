package io.muvr.em.net

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions

class MLP {

  def model(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val seed = 666
    val iterations = 20

    val distribution: UniformDistribution = new UniformDistribution(-0.5, 0.5)
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .constrainGradientToUnitNorm(true)
      .learningRate(1e-3f)
      .momentum(0.1)
      .maxNumLineSearchIterations(10)
      .list(6)
      .layer(0, new DenseLayer.Builder()
        .nIn(numInputs)
        .nOut(2500)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(2500)
        .nOut(2000)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(2000)
        .nOut(1500)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(3, new DenseLayer.Builder()
        .nIn(1500)
        .nOut(1000)
        .activation("relu")
        .weightInit(WeightInit.DISTRIBUTION)
        .dist(distribution)
        .build())
      .layer(4, new DenseLayer.Builder()
        .nIn(1000)
        .nOut(500)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .dist(distribution)
        .build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
        .nIn(500)
        .nOut(numOutputs)
        .activation("sigmoid")
        .weightInit(WeightInit.DISTRIBUTION)
        .dist(distribution)
        .build())
      .backprop(true).pretrain(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    model
  }

}
