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

  case class DataSet(labels: Labels, numInputs: Int, examples: Examples) {
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

class CuratedExerciseDataSet(directory: File) extends ExerciseDataSet {
  import ExerciseDataSet._

  private def smoothi(collection: Array[Float], width: Int): Unit = {
    def average(xs: Array[Float]): Float = {
      xs.sum / xs.length.toFloat
    }

    var x: Int = width / 2
    collection.sliding(width).foreach { neighbours => collection(x) = average(neighbours); x += 1 }
  }

  private def preprocess(samples: List[Array[Float]]): INDArray = {
    val xs = samples.map(_.apply(0)).toArray
    val ys = samples.map(_.apply(1)).toArray
    val zs = samples.map(_.apply(2)).toArray

    smoothi(xs, 5)
    smoothi(ys, 5)
    smoothi(zs, 5)

    Nd4j.create(xs ++ ys ++ zs)
  }

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
            /** convert-clip-norm the string ``s`` into a ``Float``, normalize to 20 m/s*s, clip to (-1, 1) */
            def ccn(s: String): Float = {
              val x = s.toFloat / norm
              if (x > 1) 1 else if (x < -1) -1 else x
            }
            // only take the labels for which ``labelTransform`` returns ``Some``
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
            val samples = preprocess(window)
            Some(labelVector → samples)
          } else None
        }
    })
    val examplesMatrix = Nd4j.create(examplesAndLabelVectors.size, windowSize * windowDimension)
    val labelsMatrix = Nd4j.create(examplesAndLabelVectors.size, labels.size)
    examplesAndLabelVectors.zipWithIndex.foreach {
      case ((label, example), i) ⇒
        examplesMatrix.putRow(i, example)
        labelsMatrix.putRow(i, label)
    }

    DataSet(Labels(labels), 1200, (examplesMatrix, labelsMatrix))
  }

  override def labelsAndExamples: DataSet = loadFilesInDirectory(directory) {
    case "" => None
    case "triceps-dips" => None
    case "dumbbell-bench-press" => None
    case x => Some(x)
  }

  override def exerciseVsSlacking: DataSet = loadFilesInDirectory(directory) { label ⇒
    if (label.isEmpty) Some("-") else Some("E")
  }
}
