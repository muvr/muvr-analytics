package io.muvr.em

import java.io._

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{Labels, CuratedExerciseDataSet, ExerciseDataSet}
import io.muvr.em.net.{DBN, MLP}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

object ModelTrainer extends App {

  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "arms"

  type ModelId = String
  type NewModel = (Int, Int) => MultiLayerNetwork

  def train(dataSet: DataSet, modelName: String, newModel: NewModel): MultiLayerNetwork = {
    // construct the "empty" model
    val model = newModel(dataSet.numInputs, dataSet.numOutputs)
    dataSet.examples.foreach { example =>
      // fit the examples and labels in the model
      model.fit(example._1, example._2)
    }
    model
  }

  def save(name: String, model: MultiLayerNetwork, labels: Labels): Unit = {
    import ModelPersistance._
    model.save(rootDirectory, name)
    labels.save(rootDirectory, name)
  }

  def evaluate(model: MultiLayerNetwork, labels: Labels, dataSet: DataSet): Double = {
    val (examplesMatrix, labelsMatrix) = dataSet.examples.next()
    val cm = Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
    print(cm.toPrettyString(labels))
    cm.accuracy()
  }

  def pipeline(dataSet: DataSet)(mm: (ModelId, NewModel)): Double = {
    val (id, newModel) = mm
    val model = train(dataSet, datasetName, newModel)
    save(datasetName, model, dataSet.labels)
    evaluate(model, dataSet.labels, dataSet)
  }

  val dataSet = new CuratedExerciseDataSet(
    directory = new File(s"$rootDirectory/train/$datasetName"),
    multiplier = 1)

  val models: List[(ModelId, NewModel)] = List(("dbn", DBN.newModel), ("mlp", MLP.newModel))

  models.foreach(pipeline(dataSet.labelsAndExamples))
  models.foreach(pipeline(dataSet.exerciseVsSlacking))

}
