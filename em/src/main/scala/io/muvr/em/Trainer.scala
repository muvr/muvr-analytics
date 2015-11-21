package io.muvr.em

import java.io._

import io.muvr.em.dataset.CuratedExerciseDataSetLoader
import io.muvr.em.net.MLP

object Trainer extends App {

  val dataset = new CuratedExerciseDataSetLoader(
    trainDirectory = new File("/Users/janmachacek/Tmp/labelled/core"),
    multiplier = 10)

  val (examples, labels, _) = dataset.train

  // construct the "empty" model
  val model = new MLP().model(examples.columns(), labels.columns())

  // fit the examples and labels in the model
  model.fit(examples, labels)

  val oos = new ObjectOutputStream(new FileOutputStream("/Users/janmachacek/Tmp/model.ser"))
  oos.writeObject(model)
  oos.close()

  val (s, f) = Evaluation.evaluate(model, examples, labels)
  println(s"Succeeded $s, failed $f")

}
