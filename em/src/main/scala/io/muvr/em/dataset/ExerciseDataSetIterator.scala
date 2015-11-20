package io.muvr.em.dataset

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.NDArray

trait ExerciseDataSetLoader {

  type ExamplesAndLabels = (INDArray, INDArray)

  def train: ExamplesAndLabels

  def test: ExamplesAndLabels

}

class SyntheticExerciseDataSetLoader(numClasses: Int, numExamples: Int) extends ExerciseDataSetLoader {

  override lazy val train: ExamplesAndLabels = {
    val exampleSamples = 400
    val exampleDimensions = 3

    val examples = new NDArray(numExamples, exampleSamples * exampleDimensions)
    val labels = new NDArray(numExamples, numClasses)
    (0 until numExamples).foreach { row =>
      val clazz = row % numClasses
      val exampleValues = (0 until exampleSamples).flatMap { pos =>
        val v = math.sin(clazz * pos / exampleSamples.toDouble).toFloat
        Array(v, v, v)
      }
      val example = new NDArray(exampleValues.toArray)
      val label = new NDArray((0 until numClasses).map { c => if (c == clazz) 1.toFloat else 0.toFloat }.toArray)

      labels.putRow(row, label)
      examples.putRow(row, example)
    }

    (examples, labels)
  }

  override lazy val test: ExamplesAndLabels = train

}
