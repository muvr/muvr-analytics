package io.muvr.em

import io.muvr.em.dataset.SyntheticExerciseDataSetLoader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions

object ExerciseCNN extends App {

  val seed = 666
  val batchSize = 500
  val iterations = 20

  val nChannels = 1

  val (examples, labels) = new SyntheticExerciseDataSetLoader(10, numExamples = 1000).train

  val builder = new NeuralNetConfiguration.Builder()
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

  new ConvolutionLayerSetup(builder, 3, 400, nChannels)
  val model = new MultiLayerNetwork(builder.build)
  model.init()
  model.setListeners(new ScoreIterationListener(1))
  
  model.fit(examples, labels)
  val eval = new Evaluation(labels.columns())
  (0 until examples.rows()).foreach { row =>
    val example = examples.getRow(row)
    val label = labels.getRow(row)
    println(s"Predicted ${model.predict(example)}, expected $label")
  }

}
