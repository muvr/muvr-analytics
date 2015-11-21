package io.muvr.em

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

object Evaluation {
  import Implicits._

  def evaluate(model: MultiLayerNetwork, examples: INDArray, labels: INDArray): (Int, Int) = {
    val z: (Int, Int) = (0, 0)
    (0 until examples.rows()).foldLeft(z) {
      case ((succeeded, failed), row) =>
        val example = examples.getRow(row)
        val (ei, _) = labels.getRow(row).maxf
        val (pi, _) = model.output(example).maxf
        if (ei == pi) (succeeded + 1, failed) else (succeeded, failed + 1)
    }
  }

}
