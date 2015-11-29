package io.muvr.em.dataset

/**
  * The labels
  * @param labels label names
  */
case class Labels(labels: Array[String]) extends AnyVal {
  def length: Int = labels.length
}
