package io.muvr.em

import org.nd4j.linalg.api.ndarray.INDArray

object Implicits {

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

}
