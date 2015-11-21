package io.muvr.em.net

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

class MLP {

  def model(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val seed = 666
    val iterations = 20

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .learningRate(1e-3)
      .l1(0.3).regularization(true).l2(1e-3)
      .constrainGradientToUnitNorm(true)
      .list(3)
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(500)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(500).nOut(200)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .nIn(200).nOut(numOutputs).build())
      .backprop(true)
      .pretrain(false)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    model
  }

}
