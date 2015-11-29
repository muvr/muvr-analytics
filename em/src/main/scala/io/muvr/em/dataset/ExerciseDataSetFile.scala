package io.muvr.em.dataset


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

  /**
    * Extract the labels from the CSV lines in ``lines``
    * @param lines the lines in the file
    * @return the labels
    */
  def extractLabels(lines: Array[String]): Array[String] = {
    lines.flatMap { _.split(",", -1) match {
      case Array(_, _, _, label, _, _, _) ⇒ Some(label)
      case _ ⇒ None
    }}
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
          labelTransform(label).map(label ⇒ (label, Array(ccn(x), ccn(y), ccn(z))))
        case _ ⇒
          None
      }
    }.groupBy(_._1).mapValues(_.map(_._2)).toList
  }

}
