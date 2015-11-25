package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

case class ConfusionMatrix(labelCount: Int) {
  private val entries: Array[Array[Int]] = Array.fill(labelCount)(Array.fill(labelCount)(0))

  def +=(actualLabel: Int, predictedLabel: Int): Unit = {
    entries(actualLabel)(predictedLabel) = entries(actualLabel)(predictedLabel) + 1
  }

  def toPrettyString(labelNames: List[String]): String = {
    val sb = new StringBuilder()
    val labelWidth = labelNames.map(_.length).max + 2
    implicit class LabelStringOps(s: String) {
      lazy val labelText: String = s.padTo(labelWidth, " ").take(labelWidth).mkString
    }

    val emptyLabel = "".labelText

    sb.append(emptyLabel).append(" | ")
    labelNames.foreach { label ⇒ sb.append(label.labelText) }
    sb.append("\n")
    sb.append("".padTo(sb.length, "-").mkString)
    sb.append("\n")

    labelNames.zipWithIndex.foreach { case (label, i) ⇒
      sb.append(label.labelText).append(" | ")
      labelNames.indices.foreach { j ⇒
        val v = entries(i)(j)
        sb.append(v.toString.labelText)
      }
      sb.append("\n")
    }
    sb.append("\n")

    sb.toString
  }


}

object Evaluation {
  import Implicits._

  def evaluate(model: MultiLayerNetwork, examples: INDArray, labels: INDArray): ConfusionMatrix = {
    val cm = ConfusionMatrix(labels.columns())

    (0 until examples.rows()).par.foreach { row =>
        val example = examples.getRow(row)
        val (ai, _) = labels.getRow(row).maxf
        val (pi, _) = model.output(example).maxf

        cm += (ai, pi)
    }

    cm
  }

}
