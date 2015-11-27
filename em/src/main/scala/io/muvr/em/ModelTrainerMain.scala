package io.muvr.em

import java.io.File

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{Labels, ExerciseDataSetFile}
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.factory.Nd4j

object ModelTrainerMain {

  def main(args: Array[String]): Unit = {
    val master   = "local"
    val model    = "arms"
    val labelTransform: (String ⇒ Option[String]) = {
      case "" ⇒ None
      case "triceps-dips" ⇒ None
      case "dumbbell-bench-press" ⇒ None
      case x ⇒ Some(x)
    }

    val name = "Train"
    val conf = new SparkConf().
      setMaster(master).
      setAppName(name).
      set("spark.app.id", name)
    val sc = new SparkContext(conf)
    val path = s"/Users/janmachacek/Muvr/muvr-open-training-data/train/$model"
    val outputPath = new File("/Users/janmachacek/Muvr/muvr-open-training-data/models")
    outputPath.mkdirs()

    val fileContents = sc.wholeTextFiles(path)
    // each entry contains the label => Array of x,y,z acceleration values
    // Array[Map[String, Array[Array[Float]]]]
    val fileLabelsAndExamples = fileContents.map { case (_, text) ⇒ ExerciseDataSetFile.parse(text.split("\n"))(labelTransform) }.collect()

    // all labels is the distinct set of all keys in all the maps
    val labels = fileLabelsAndExamples.flatMap(_.map(_._1)).distinct
    val windowSize = 400
    val windowStep = 50
    val windowDimension = 3
    val examplesAndLabelVectors = fileLabelsAndExamples.flatMap(_.flatMap {
      case (label, samples) ⇒
        val labelIndex = labels.indexOf(label)
        val labelVector = Nd4j.create(labels.indices.map { l ⇒ if (l == labelIndex) 1.toFloat else 0.toFloat }.toArray)
        samples.sliding(windowSize, windowStep).flatMap { window ⇒
          if (window.length == windowSize) {
            val samples = window.flatten
            Some(labelVector → Nd4j.create(samples))
          } else None
        }
    })

    val examplesMatrix = Nd4j.create(examplesAndLabelVectors.length, windowSize * windowDimension)
    val labelsMatrix = Nd4j.create(examplesAndLabelVectors.length, labels.length)
    examplesAndLabelVectors.zipWithIndex.foreach {
      case ((label, example), i) ⇒
        examplesMatrix.putRow(i, example)
        labelsMatrix.putRow(i, label)
    }

    val ds = DataSet(Labels(labels.toList), windowSize * windowDimension, (examplesMatrix, labelsMatrix))
    val best = new ModelTrainer(new ModelPersistor(outputPath)).execute(model, ds, ds)
    println(best)
  }
}
