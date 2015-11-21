package io.muvr.em

import java.io.File

import io.muvr.em.dataset.{CuratedExerciseDataSetLoader, SyntheticExerciseDataSetLoader}
import io.muvr.em.net.MLP
import org.nd4j.linalg.api.ndarray.INDArray

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

//  val dataset = new SyntheticExerciseDataSetLoader(10, numExamples = 50000)
  val dataset = new CuratedExerciseDataSetLoader(
    trainDirectory = new File("/Users/janmachacek/Tmp/labelled/x"),
    multiplier = 20)

  val (examples, labels) = dataset.train

  // construct the "empty" model
  val model = new MLP().model(examples.columns(), labels.columns())

  // fit the examples and labels in the model
  model.fit(examples, labels)

  // evaluate
  val z: (Int, Int) = (0, 0)
  val (s, f) = (0 until examples.rows()).foldLeft(z) {
    case ((succeeded, failed), row) =>
      val example = examples.getRow(row)
      val (ei, _) = labels.getRow(row).maxf
      val (pi, _) = model.output(example).maxf
      if (ei == pi) (succeeded + 1, failed) else (succeeded, failed + 1)
  }
  println(s"Succeeded $s, failed $f")

}
