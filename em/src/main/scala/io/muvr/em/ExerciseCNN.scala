package io.muvr.em

import io.muvr.em.dataset.SyntheticExerciseDataSetLoader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


object ExerciseCNN extends App {
  implicit class INDArrayOps(x: INDArray) {
    def maxf: (Int, Float) = {
      val zero: (Int, Float) = (0, 0)
      (0 until x.columns()).foldLeft(zero) {
        case ((i, v), column) =>
          val cv = x.getFloat(column)
          if (cv > v) (column, cv) else (i, v)
      }
    }
  }

  val seed = 666
  val batchSize = 500
  val iterations = 20

  val nChannels = 1

  val (examples, labels) = new SyntheticExerciseDataSetLoader(10, numExamples = 10000).train

  val builder = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .learningRate(1e-3)
    .l1(0.3).regularization(true).l2(1e-3)
    .constrainGradientToUnitNorm(true)
    .list(3)
    .layer(0, new DenseLayer.Builder().nIn(1200).nOut(500)
      .activation("tanh")
      .weightInit(WeightInit.XAVIER)
      .build())
    .layer(1, new DenseLayer.Builder().nIn(500).nOut(200)
      .activation("tanh")
      .weightInit(WeightInit.XAVIER)
      .build())
    .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
      .weightInit(WeightInit.XAVIER)
      .activation("softmax")
      .nIn(200).nOut(labels.columns()).build())
    .backprop(true)
    .pretrain(false)

  /*
  .seed(seed)
  .batchSize(batchSize)
  .iterations(iterations)
  .constrainGradientToUnitNorm(true)
  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
  .list(3)
  .layer(0, new ConvolutionLayer.Builder(3, 200)
    .stride(2,2)
    .nIn(nChannels)
    .nOut(6)
    .weightInit(WeightInit.XAVIER)
    .activation("relu")
    .build())
  .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(1, 2))
    .build())
  .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .nOut(labels.columns())
    .weightInit(WeightInit.XAVIER)
    .activation("softmax")
    .build())
  .backprop(true)
  .pretrain(false)
  */

  //new ConvolutionLayerSetup(builder, 3, 400, nChannels)
  val model = new MultiLayerNetwork(builder.build)
  model.init()
  model.setListeners(new ScoreIterationListener(1))

  model.fit(examples, labels)
  val eval = new Evaluation(labels.columns())
  (0 until examples.rows()).foreach { row =>
    val example = examples.getRow(row)
    val expected = labels.getRow(row).maxf
    val predicted = model.output(example).maxf

    println(s"Predicted $predicted, expected $expected")
  }

}
