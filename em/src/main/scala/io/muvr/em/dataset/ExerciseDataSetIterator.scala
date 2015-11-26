package io.muvr.em.dataset

import java.io.File

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory._

import scala.io.Source

/**
  * The labels
  * @param labels label names
  */
case class Labels(labels: List[String]) extends AnyVal

/**
  * EDS type definitions
  */
object ExerciseDataSet {
  /**
    * EAL is matrix of examples with matching rows in labels and label names
    */
  type ExamplesAndLabels = (INDArray, INDArray, Labels)

  /**
    * The examples
    */
  type Examples = (INDArray, INDArray)

  case class DataSet(labels: Labels, numInputs: Int, examples: Iterator[Examples]) {
    lazy val numOutputs: Int = labels.labels.length
  }

}

/**
  * Implementations of EDS must provide iterator over EALs
  */
trait ExerciseDataSet {
  import ExerciseDataSet._

  /**
    * The iterator of distinct exercises
    * @return the I of EAL
    */
  def labelsAndExamples: DataSet

  /**
    * The iterator of exercise vs. slacking
    * @return the I of EAL
    */
  def exerciseVsSlacking: DataSet

}

class SyntheticExerciseDataSet(numClasses: Int, numExamples: Int) extends ExerciseDataSet {
  import ExerciseDataSet._

  override lazy val labelsAndExamples: DataSet = {
    val exampleSamples = 400
    val exampleDimensions = 3

    val examples = Nd4j.create(numExamples, exampleSamples * exampleDimensions)
    val labels = Nd4j.create(numExamples, numClasses)
    (0 until numExamples).foreach { row =>
      val clazz = row % numClasses
      val exampleValues = (0 until exampleSamples).flatMap { pos =>
        val v = math.sin(clazz * pos / exampleSamples.toDouble) + (math.random * 0.1)
        Array(v.toFloat, v.toFloat, v.toFloat)
      }
      val example = Nd4j.create(exampleValues.toArray)
      val label = Nd4j.create((0 until numClasses).map { c => if (c == clazz) 1.toFloat else 0.toFloat }.toArray)

      labels.putRow(row, label)
      examples.putRow(row, example)
    }

    DataSet(Labels((0 until labels.rows()).map(_.toString).toList), 1200, Iterator((examples, labels)))
  }

  override def exerciseVsSlacking: DataSet = ???
}

class CuratedExerciseDataSet(directory: File, multiplier: Int = 1) extends ExerciseDataSet {
  import ExerciseDataSet._

  private def loadFilesInDirectory(directory: File)
                                  (labelTransform: String ⇒ Option[String]): DataSet = {
    val windowSize = 400
    val windowStep = 50
    val windowDimension = 3
    val norm: Float = 2

    val filesAndLabels = directory.listFiles().toList.map { file ⇒
      // each file contains potentially more than one label
      Source.fromFile(file).getLines().toList.flatMap { line ⇒
        line.split(",", -1) match {
          case Array(x, y, z, label, _, _, _) ⇒
            def ccn(s: String): Float = {
              val x = s.toFloat / norm
              if (x > 1) 1 else if (x < -1) -1 else x
            }
            labelTransform(label).map(label ⇒ label → Array(ccn(x), ccn(y), ccn(z)))
          case _ ⇒
            None
        }
      }.groupBy(_._1).mapValues(_.map(_._2))
    }

    val labels = filesAndLabels.flatMap(_.keys).distinct
    val examplesAndLabelVectors = filesAndLabels.par.flatMap(_.toList.flatMap {
      case (label, samples) ⇒
        val labelIndex = labels.indexOf(label)
        val labelVector = Nd4j.create(labels.indices.map { l ⇒ if (l == labelIndex) 1.toFloat else 0.toFloat }.toArray)
        samples.sliding(windowSize, windowStep).flatMap { window ⇒
          if (window.size == windowSize) {
            (0 until multiplier).map { i ⇒
              val samples = window.flatten.map { v ⇒ if (i == 0) v else v + (math.random * 0.1).toFloat }
              labelVector → Nd4j.create(samples.toArray)
            }
          } else Nil
        }
    })
    val examplesMatrix = Nd4j.create(examplesAndLabelVectors.size, windowSize * windowDimension)
    val labelsMatrix = Nd4j.create(examplesAndLabelVectors.size, labels.size)
    examplesAndLabelVectors.zipWithIndex.foreach {
      case ((label, example), i) ⇒
        examplesMatrix.putRow(i, example)
        labelsMatrix.putRow(i, label)
    }

    DataSet(Labels(labels), 1200, Iterator((examplesMatrix, labelsMatrix)))
  }

  override def labelsAndExamples: DataSet = loadFilesInDirectory(directory) { label ⇒
    if (label.isEmpty) None else Some(label)
  }

  override def exerciseVsSlacking: DataSet = loadFilesInDirectory(directory) { label ⇒
    if (label.isEmpty) Some("-") else Some("E")
  }
}
