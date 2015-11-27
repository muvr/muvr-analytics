package io.muvr.em

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Convenience functions for ``INDArray``s
  */
object INDArrayImplicits {

  /**
    * Adds convenience methods to ``INDArray``s
    * @param x the underlying array
    */
  implicit class INDArrayOps(x: INDArray) {

    /**
      * Returns the index and value of the maximum element of a _vector_ ``x``
      * @return the index and value of the max
      */
    def maxf: (Int, Float) = {
      val zero: (Int, Float) = (0, 0)
      (0 until x.columns()).foldLeft(zero) {
        case ((i, v), column) ⇒
          val cv = x.getFloat(column)
          if (cv > v) (column, cv) else (i, v)
      }
    }
  }

}
