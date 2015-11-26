package io.muvr.em

import java.io.{BufferedWriter, OutputStream}

import io.muvr.em.dataset.Labels
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * The confusion matrix
  * @param labelCount the number of labels
  */
case class ConfusionMatrix(labelCount: Int) {
  private val entries: Array[Array[Int]] = Array.fill(labelCount)(Array.fill(labelCount)(0))
  private var predictions: Int = 0
  private var truePositives: Int = 0
  private var falsePositives: Int = 0

  /**
    * Adds a predicted vs. actual label prediction
    * @param actualLabel the actual label
    * @param predictedLabel the predicted label
    */
  def +=(actualLabel: Int, predictedLabel: Int): Unit = {
    entries(actualLabel)(predictedLabel) = entries(actualLabel)(predictedLabel) + 1
    predictions += 1
    if (actualLabel == predictedLabel) truePositives += 1
    else falsePositives += 1
  }

  /**
    * Computes the accuracy
    * @return the accuracy
    */
  def accuracy(): Double = truePositives.toDouble / predictions.toDouble

  /**
    * Save this CM into a CSV file
    * @param labels the labels
    * @param out the output buffer
    */
  def saveAsCsv(labels: Labels, out: BufferedWriter): Unit = {
    out.write("-,")
    labels.labels.zipWithIndex.foreach { case (label, i) ⇒
      out.write(s""""$label"""")
      if (i < labels.labels.length - 1) out.write(",")
    }
    out.write("\n")
    labels.labels.zipWithIndex.foreach { case (actual, i) ⇒
      out.write(s""""$actual""""); out.write(",")
      labels.labels.indices.foreach { j ⇒
        val v = entries(i)(j)
        out.write(v.toString)
        if (j < labels.labels.length - 1) out.write(",")
      }
      out.write("\n")
    }
    out.close()
  }

  /**
    * Prints confusion matrix
    * @param labels the label names
    * @return the confusion matrix string
    */
  def toPrettyString(labels: Labels): String = {
    val sb = new StringBuilder()
    val labelWidth = math.max(labels.labels.map(_.length).max + 2, 10)
    implicit class LabelStringOps(s: String) {
      lazy val labelText: String = s.padTo(labelWidth, " ").take(labelWidth).mkString
    }

    val emptyLabel = "".labelText

    sb.append(emptyLabel).append(" | ")
    labels.labels.foreach { label ⇒ sb.append(label.labelText) }
    sb.append("\n")
    sb.append("".padTo(sb.length, "-").mkString)
    sb.append("\n")

    labels.labels.zipWithIndex.foreach { case (label, i) ⇒
      sb.append(label.labelText).append(" | ")
      labels.labels.indices.foreach { j ⇒
        val v = entries(i)(j)
        sb.append(v.toString.labelText)
      }
      sb.append("\n")
    }
    sb.append("\n")
    sb.append("\n")
    sb.append(s"Accuracy = ${accuracy()}\n")

    sb.toString
  }

}

object Evaluation {
  import Implicits._

  def evaluate(model: MultiLayerNetwork, examples: INDArray, labels: INDArray): ConfusionMatrix = {
    val cm = ConfusionMatrix(labels.columns())

    (0 until examples.rows()).foreach { row ⇒
        val example = examples.getRow(row)
        val (ai, _) = labels.getRow(row).maxf
        val (pi, _) = model.output(example).maxf

        cm += (ai, pi)
    }

    cm
  }

}
