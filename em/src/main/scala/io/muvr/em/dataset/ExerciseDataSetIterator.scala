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
    * The examples
    */
  type Examples = (INDArray, INDArray)

  /**
    * The data set containing the labels
    * @param labels the labels
    * @param numInputs number of inputs for the ANN
    * @param examples the examples and labels
    */
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

object ExerciseDataSetFile {

  private def smoothi(collection: Array[Float], width: Int): Unit = {
    def average(xs: Array[Float]): Float = {
      xs.sum / xs.length.toFloat
    }

    var x: Int = width / 2
    collection.sliding(width).foreach { neighbours => collection(x) = average(neighbours); x += 1 }
  }

  def preprocessi(samples: Array[Array[Float]]): Unit = {
    val xs = samples.map(_.apply(0))
    val ys = samples.map(_.apply(1))
    val zs = samples.map(_.apply(2))

    smoothi(xs, 5)
    smoothi(ys, 5)
    smoothi(zs, 5)
  }

  def parse(lines: Array[String])(labelTransform: String ⇒ Option[String]): List[(String, Array[Array[Float]])] = {
    /** convert-clip-norm the string ``s`` into a ``Float``, normalize to 20 m/s*s, clip to (-1, 1) */
    def ccn(s: String): Float = {
      val norm: Float = 2
      val x = s.toFloat / norm
      if (x > 1) 1 else if (x < -1) -1 else x
    }

    lines.flatMap { _.split(",", -1) match {
        case Array(x, y, z, label, _, _, _) ⇒
          // only take the labels for which ``labelTransform`` returns ``Some``
          labelTransform(label).map(label ⇒ label → Array(ccn(x), ccn(y), ccn(z)))
        case _ ⇒
          None
      }
    }.groupBy(_._1).mapValues(_.map(_._2)).toList
  }



}

//class CuratedExerciseDataSet(directory: File) extends ExerciseDataSet {
//  import ExerciseDataSet._
//
//  private def smoothi(collection: Array[Float], width: Int): Unit = {
//    def average(xs: Array[Float]): Float = {
//      xs.sum / xs.length.toFloat
//    }
//
//    var x: Int = width / 2
//    collection.sliding(width).foreach { neighbours => collection(x) = average(neighbours); x += 1 }
//  }
//
//  private def preprocessi(samples: Array[Array[Float]]): Unit = {
//    val xs = samples.map(_.apply(0))
//    val ys = samples.map(_.apply(1))
//    val zs = samples.map(_.apply(2))
//
//    smoothi(xs, 5)
//    smoothi(ys, 5)
//    smoothi(zs, 5)
//  }
//
//  private def loadFilesInDirectory(directory: File)
//                                  (labelTransform: String ⇒ Option[String]): DataSet = {
//    val windowSize = 400
//    val windowStep = 50
//    val windowDimension = 3
//    val norm: Float = 2
//
//    val filesAndLabels = directory.listFiles().toList.map { file ⇒
//      // each file contains potentially more than one label
//      val parsed = ExerciseDataSetFile.parse(Source.fromFile(file).getLines().toArray)(labelTransform)
//      parsed.foreach(x ⇒ ExerciseDataSetFile.preprocessi(x._2))
//      parsed
//    }
//
//    val labels = filesAndLabels.flatMap(_.map(_._1)).distinct
//    val examplesAndLabelVectors = filesAndLabels.par.flatMap(_.flatMap {
//      case (label, samples) ⇒
//        val labelIndex = labels.indexOf(label)
//        val labelVector = Nd4j.create(labels.indices.map { l ⇒ if (l == labelIndex) 1.toFloat else 0.toFloat }.toArray)
//        samples.sliding(windowSize, windowStep).flatMap { window ⇒
//          if (window.length == windowSize) {
//            preprocessi(window)
//            Some(labelVector → Nd4j.create(samples.flatten))
//          } else None
//        }
//    })
//    val examplesMatrix = Nd4j.create(examplesAndLabelVectors.size, windowSize * windowDimension)
//    val labelsMatrix = Nd4j.create(examplesAndLabelVectors.size, labels.size)
//    examplesAndLabelVectors.zipWithIndex.foreach {
//      case ((label, example), i) ⇒
//        examplesMatrix.putRow(i, example)
//        labelsMatrix.putRow(i, label)
//    }
//
//    DataSet(Labels(labels), 1200, (examplesMatrix, labelsMatrix))
//  }
//
//  override def labelsAndExamples: DataSet = loadFilesInDirectory(directory) {
//    case "" => None
//    case "triceps-dips" => None
//    case "dumbbell-bench-press" => None
//    case x => Some(x)
//  }
//
//  override def exerciseVsSlacking: DataSet = loadFilesInDirectory(directory) { label ⇒
//    if (label.isEmpty) Some("-") else Some("E")
//  }
//}
