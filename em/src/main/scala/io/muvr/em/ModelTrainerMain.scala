package io.muvr.em

import java.io.File

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{Labels, ExerciseDataSetFile}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.factory.Nd4j

object ModelTrainerMain {

  /**
    * Transform RDD of CSV Muvr file contents into a DataSet
    * @param files the file contents
    * @param labelTransform the label transform function
    * @return the local ``DataSet``
    */
  private def dataSet(files: RDD[(String, String)])(labelTransform: String ⇒ Option[String]): DataSet = {
    val fileLabelsAndExamples = files.map { case (_, text) ⇒ ExerciseDataSetFile.parse(text.split("\n"))(labelTransform) }.collect()

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
            Some((labelVector, Nd4j.create(samples)))
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

    DataSet(Labels(labels.toList), windowSize * windowDimension, (examplesMatrix, labelsMatrix))
  }

  /**
    * Starts the Spark app
    * @param args the args
    */
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
    val trainPath = s"/Users/janmachacek/Muvr/muvr-open-training-data/train/$model"
    val testPath = s"/Users/janmachacek/Muvr/muvr-open-training-data/train/$model"
    val outputPath = new File("/Users/janmachacek/Muvr/muvr-open-training-data/models"); outputPath.mkdirs()
    val trainer = new ModelTrainer(new ModelPersistor(outputPath))

    val train = dataSet(sc.wholeTextFiles(trainPath))(labelTransform)
    val test  = dataSet(sc.wholeTextFiles(testPath))(labelTransform)
    val best  = trainer.execute(model, train, test)
    println(best)
  }
}
