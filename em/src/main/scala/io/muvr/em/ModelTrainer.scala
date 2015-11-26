package io.muvr.em

import java.io._

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{Labels, CuratedExerciseDataSet, ExerciseDataSet}
import io.muvr.em.models.{DBN, MLP}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

object ModelTrainer extends App {

  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "arms"

  def train(dataSet: DataSet, modelName: String, modelConstructor: Model.Constructor): MultiLayerNetwork = {
    // construct the "empty" model
    val model = modelConstructor(dataSet.numInputs, dataSet.numOutputs)
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    // fit the examples and labels in the model
    model.fit(examplesMatrix, labelsMatrix)
    model
  }

  def save(name: String, model: MultiLayerNetwork, labels: Labels): Unit = {
    import ModelPersistance._
    model.save(rootDirectory, name)
    labels.save(rootDirectory, name)
  }

  def evaluate(model: MultiLayerNetwork, labels: Labels, dataSet: DataSet): Double = {
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    val cm = Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
    print(cm.toPrettyString(labels))
    cm.accuracy()
  }

  def pipeline(trainDataSet: DataSet, testDataSet: DataSet)(model: Model): (Model.Id, Double) = {
    val m = train(trainDataSet, datasetName, model.modelConstructor)
    save(datasetName, m, trainDataSet.labels)
    val result = evaluate(m, testDataSet.labels, testDataSet)
    (model.id, result)
  }

  val dataSet = new CuratedExerciseDataSet(
    directory = new File(s"$rootDirectory/train/$datasetName"),
    multiplier = 1)

  val models: List[Model] = List(DBN.model, MLP.model)

  val result = models.map(pipeline(dataSet.labelsAndExamples, dataSet.labelsAndExamples))
  println(result)
  // models.foreach(pipeline(dataSet.exerciseVsSlacking, dataSet.exerciseVsSlacking))

}
