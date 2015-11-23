package io.muvr.em

import java.io._

import io.muvr.em.dataset.{CuratedExerciseDataSet, ExerciseDataSet}
import io.muvr.em.net.MLP
import org.nd4j.linalg.factory.Nd4j

object Trainer extends App {

  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "arms"

  def trainAndSave(dataset: ExerciseDataSet.ExamplesAndLabels, modelName: String): Unit = {
    val (examplesMatrix, labelsMatrix, labelNames) = dataset

    // construct the "empty" model
    val model = new MLP().model(examplesMatrix.columns(), labelsMatrix.columns())

    // fit the examples and labels in the model
    model.fit(examplesMatrix, labelsMatrix)

    val labelsOutputStream = new FileOutputStream(s"$rootDirectory/models/$modelName.labels")
    labelsOutputStream.write(labelNames.mkString("\n").getBytes("UTF-8"))
    labelsOutputStream.close()

    Nd4j.write(model.params(), new DataOutputStream(new FileOutputStream(s"$rootDirectory/models/$modelName.params")))
    val modelConfOutputStream = new FileOutputStream(s"$rootDirectory/models/$modelName.conf")
    modelConfOutputStream.write(model.conf().toYaml.getBytes("UTF-8"))
    modelConfOutputStream.close()

    val modelOutputStream = new ObjectOutputStream(new FileOutputStream(s"$rootDirectory/models/$modelName.model"))
    modelOutputStream.writeObject(model)
    modelOutputStream.close()

    val (s, f) = Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
    println(s"Succeeded $s, failed $f")
  }

  val dataset = new CuratedExerciseDataSet(
    directory = new File(s"$rootDirectory/train/$datasetName"),
    multiplier = 10)

  trainAndSave(dataset.labelsAndExamples, datasetName)
  trainAndSave(dataset.exerciseVsSlacking, s"$datasetName-es")

}
