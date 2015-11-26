package io.muvr.em

import java.io._

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{CuratedExerciseDataSet, Labels}
import io.muvr.em.models.{DBN, MLP}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import scala.tools.nsc.ConsoleWriter

object ModelTrainer extends App {

  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "arms"

  /**
    * Train the model created by ``modelConstructor`` on the given ``dataSet``, with some specific ``modelName``
    * @param dataSet the training data set
    * @param modelName the model name
    * @param modelConstructor the model constructor
    * @return the MLN
    */
  def train(dataSet: DataSet, modelName: String, modelConstructor: Model.Constructor): MultiLayerNetwork = {
    // construct the "empty" model
    val model = modelConstructor(dataSet.numInputs, dataSet.numOutputs)
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    // fit the examples and labels in the model
    model.fit(examplesMatrix, labelsMatrix)
    model
  }

  /**
    * Saves the model configuration and parameters, and labels
    *
    * @param modelName the model name
    * @param model the model
    * @param labels the labels
    */
  def save(modelName: String, model: MultiLayerNetwork, cm: ConfusionMatrix, labels: Labels): Unit = {
    import ModelPersistance._
    model.save(rootDirectory, modelName)
    labels.save(rootDirectory, modelName)
    cm.save(rootDirectory, modelName, labels)
  }

  /**
    * Evaluates the given ``model`` on the test ``dataSet`` with human-readable ``labels``
    * @param model the (trained) model
    * @param labels the label strings
    * @param dataSet the test data set
    * @return the accuracy
    */
  def evaluate(model: MultiLayerNetwork, labels: Labels, dataSet: DataSet): ConfusionMatrix = {
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
  }

  /**
    * Model pipeline is to construct a model, train on the train data set, evaluate on the test data set, save
    * and return model identity and accuracy
    *
    * @param trainDataSet the train data set
    * @param testDataSet the test data set
    * @param model the model metadata
    * @return model id and accuracy
    */
  def pipeline(tag: String, trainDataSet: DataSet, testDataSet: DataSet)(model: Model): (Model.Id, Double) = {
    val m = train(trainDataSet, datasetName, model.modelConstructor)
    val cm = evaluate(m, testDataSet.labels, testDataSet)
    save(s"$datasetName-$tag-${model.id}", m, cm, trainDataSet.labels)

    println(model.id)
    println(cm.toPrettyString(testDataSet.labels))
    println()

    (model.id, cm.accuracy())
  }

  val dataSet = new CuratedExerciseDataSet(directory = new File(s"$rootDirectory/train/$datasetName"))

  val models: List[Model] = List(MLP.shallowModel, MLP.shallowModel, MLP.shallowModel, MLP.shallowModel, DBN.model)

  val bem = models.map(pipeline("E", dataSet.labelsAndExamples, dataSet.labelsAndExamples)).maxBy(_._2)
  val bsm = models.map(pipeline("S", dataSet.exerciseVsSlacking, dataSet.exerciseVsSlacking)).maxBy(_._2)

  println(bem)
  println(bsm)

}
