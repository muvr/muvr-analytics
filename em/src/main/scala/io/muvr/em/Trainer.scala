package io.muvr.em

import java.io._

import io.muvr.em.dataset.CuratedExerciseDataSet
import io.muvr.em.net.MLP

object Trainer extends App {

  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "core"

  val dataset = new CuratedExerciseDataSet(
    directory = new File(s"$rootDirectory/train/$datasetName"),
    multiplier = 10)

  val (examplesMatrix, labelsMatrix, labelNames) = dataset.labelsAndExamples

  // construct the "empty" model
  val model = new MLP().model(examplesMatrix.columns(), labelsMatrix.columns())

  // fit the examples and labels in the model
  model.fit(examplesMatrix, labelsMatrix)

  val labelsOutputStream = new FileOutputStream(s"$rootDirectory/models/$datasetName.labels")
  labelsOutputStream.write(labelNames.mkString("\n").getBytes("UTF-8"))
  labelsOutputStream.close()

  val modelOutputStream = new ObjectOutputStream(new FileOutputStream(s"$rootDirectory/models/$datasetName.model"))
  modelOutputStream.writeObject(model)
  modelOutputStream.close()

  val (s, f) = Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
  println(s"Succeeded $s, failed $f")

}
