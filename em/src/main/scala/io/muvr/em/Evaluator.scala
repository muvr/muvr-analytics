package io.muvr.em

import java.io._

import io.muvr.em.dataset.CuratedExerciseDataSetLoader
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

object Evaluator extends App {
  val dataset = new CuratedExerciseDataSetLoader(
    trainDirectory = new File("/Users/janmachacek/Tmp/labelled/core"),
    multiplier = 10)

  val (examples, labels) = dataset.train

  val ois = new ObjectInputStream(new FileInputStream("/Users/janmachacek/Tmp/model.ser"))
  val model = ois.readObject().asInstanceOf[MultiLayerNetwork]
  ois.close()

  val (s, f) = Evaluation.evaluate(model, examples, labels)
  println(s"Succeeded $s, failed $f")
}
