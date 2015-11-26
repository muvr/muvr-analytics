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
  private var truePositives: Map[Int, Int] = Map()
  private var trueNegatives: Map[Int, Int] = Map()
  private var falsePositives: Map[Int, Int] = Map()
  private var falseNegatives: Map[Int, Int] = Map()

  implicit class MapUpdates[K, V](m: Map[K, V]) {

    def addOne(k: K)(implicit z: Numeric[V]): Map[K, V] = {
      m + ((k, m.get(k).map(z.plus(z.one, _)).getOrElse(z.zero)))
    }

  }

  /**
    * Adds a predicted vs. actual label prediction
    * @param actualLabel the actual label
    * @param predictedLabel the predicted label
    */
  def +=(actualLabel: Int, predictedLabel: Int): Unit = {
    entries(actualLabel)(predictedLabel) = entries(actualLabel)(predictedLabel) + 1
    predictions += 1
    if (actualLabel == predictedLabel) {
      truePositives = truePositives.addOne(actualLabel)
      (0 until labelCount).foreach { label ⇒
        if (label != predictedLabel) {
          trueNegatives = trueNegatives.addOne(label)
        }
      }
    } else {
      falseNegatives = falseNegatives.addOne(actualLabel)
      falsePositives = falsePositives.addOne(predictedLabel)
      (0 until labelCount).foreach { label ⇒
        if (label != actualLabel && label != predictedLabel) {
          trueNegatives = trueNegatives.addOne(label)
        }
      }
    }
  }

  /**
    * Computes overall thing by summing over all labels, taking only true positives into account
    * @param f the function to apply to each label
    * @return the overall measure
    */
  private def overall(f: Int ⇒ Option[Double]): Double = {
    val z: (Double, Double) = (0, 0)
    val (ov, lc) = (0 until labelCount).foldLeft(z) { case ((v, l), label) ⇒
      val v2 = v + f(label).getOrElse(0.0)
      if (truePositives.getOrElse(label, 0) > 0) {
        (v2, l + 1)
      } else {
        (v2, l)
      }
    }
    ov / lc
  }

  /**
    * Computes the accuracy
    * @return the accuracy
    */
  def accuracy(): Double = truePositives.size / predictions.toDouble

  /**
    * Computes pecision for the given label
    * @param label the label
    * @return the precision
    */
  def precision(label: Int): Option[Double] = {
    for {
      tpCount ← truePositives.get(label)
      fpCount ← falsePositives.get(label)
      if tpCount > 0
    } yield tpCount.toDouble / (tpCount + fpCount).toDouble
  }

  /**
    * Computes overall precision
    * @return the overall precision
    */
  def precision(): Double = overall(precision)

  /**
    * Computes recall for the given label
    * @param label the label
    * @return the recall
    */
  def recall(label: Int): Option[Double] = {
    for {
      tpCount ← truePositives.get(label)
      fnCount ← falseNegatives.get(label)
      if tpCount > 0
    } yield tpCount.toDouble / (tpCount + fnCount).toDouble
  }

  /**
    * Computes the overall recall
    * @return the recall
    */
  def recall(): Double = overall(recall)

  /**
    * Computes the F1 score for the label
    * @param label the label
    * @return the F1
    */
  def f1(label: Int): Option[Double] = {
    for {
      prec ← precision(label)
      rec = recall()
      if prec > 0 && recall > 0
    } yield 2.0 * ((prec * rec) / (prec + rec))
  }

  /**
    * Computes the overall F1 score
    * @return the F1 score
    */
  def f1(): Double = {
    val prec = precision()
    val rec = recall()
    if (prec == 0 || rec == 0) 0
    else 2.0 * (precision * recall / (precision + recall))
  }

  /**
    * Save this CM into a CSV file
    * @param labels the labels
    * @param out the output buffer
    */
  def saveAsCsv(labels: Labels, out: BufferedWriter): Unit = {
    out.write("-,")
    labels.labels.zipWithIndex.foreach { case (label, i) ⇒
      out.write( s""""$label"""")
      if (i < labels.labels.length - 1) out.write(",")
    }
    out.write("\n")
    labels.labels.zipWithIndex.foreach { case (actual, i) ⇒
      out.write( s""""$actual"""");
      out.write(",")
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
    sb.append(s"Precision = ${precision()}\n")
    sb.append(s"Recall = ${recall()}\n")
    sb.append(s"F1 = ${f1()}\n")

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

      cm +=(ai, pi)
    }

    cm
  }

}
