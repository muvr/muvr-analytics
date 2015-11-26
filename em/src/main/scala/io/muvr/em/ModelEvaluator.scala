package io.muvr.em
/*
import java.io._

import io.muvr.em.dataset.CuratedExerciseDataSet
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import scala.io.Source

object ModelEvaluator extends App {
  val rootDirectory = "/Users/janmachacek/Muvr/muvr-open-training-data"
  val datasetName = "core"

  val dataset = new CuratedExerciseDataSet(
    directory = new File(s"$rootDirectory/test/$datasetName"),
    multiplier = 10)

  println("Loading dataset...")
  val (examplesMatrix, labelsMatrix, _) = dataset.labelsAndExamples

  println("Loading model...")
  val modelInputStream = new ObjectInputStream(new FileInputStream(s"$rootDirectory/models/$datasetName.model"))
  val model = modelInputStream.readObject().asInstanceOf[MultiLayerNetwork]
  modelInputStream.close()

  val labelNames = Source.fromInputStream(new FileInputStream(s"$rootDirectory/models/$datasetName.labels")).getLines().toList

  println("Evaluating...")
  val cm = Evaluation.evaluate(model, examplesMatrix, labelsMatrix)
  print(cm.toPrettyString(labelNames))
}
*/