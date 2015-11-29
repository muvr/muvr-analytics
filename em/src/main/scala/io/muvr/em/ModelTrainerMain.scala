package io.muvr.em

import java.io.File

import io.muvr.em.dataset.{Labels, ExerciseDataSetFile}
import io.muvr.em.model.MLP
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object ModelTrainerMain {
  private val numInputs = 1200
  private val modelTemplates: List[ModelTemplate] = List.fill(1)(MLP.shallowModel)

  private def parse(files: RDD[(String, String)], labelTransform: String ⇒ Option[String]): (Labels, RDD[(INDArray, INDArray)]) = {
    val windowStep = 50
    // parse all files
    val parsed = files.map { case (_, text) ⇒ ExerciseDataSetFile.parse(text.split("\n"))(labelTransform) }
    // extract the labels
    val labelNames = parsed.flatMap(_.map(_._1)).collect().distinct
    // extract the examples and labels
    val examplesAndLabels = parsed.flatMap(_.flatMap {
      case (label, samples) ⇒
        val labelIndex = labelNames.indexOf(label)
        val labelVector = Nd4j.create(labelNames.indices.map { l ⇒ if (l == labelIndex) 1.toFloat else 0.toFloat }.toArray)
        samples.sliding(400, windowStep).flatMap { window ⇒
          if (window.length == 400) {
            val samples = window.flatten
            Some((Nd4j.create(samples), labelVector))
          } else None
        }
    })

    (Labels(labelNames), examplesAndLabels)
  }

  private def batchToExamplesAndLabelsMatrix(batch: Seq[(INDArray, INDArray)]): (INDArray, INDArray) = {
    val (_, labels) = batch.head
    val examplesMatrix = Nd4j.create(batch.length, numInputs)
    val labelsMatrix = Nd4j.create(batch.length, labels.columns())
    batch.zipWithIndex.foreach { case ((example, label), i) ⇒
      examplesMatrix.putRow(i, example)
      labelsMatrix.putRow(i, label)
    }
    (examplesMatrix, labelsMatrix)
  }

  private def x(trainFiles: RDD[(String, String)], testFiles: RDD[(String, String)], labelTransform: String ⇒ Option[String],
               persistor: ModelPersistor)(modelTemplate: ModelTemplate) = {
    val batchSize = 50000
    val (testLabels, testExamplesAndLabels) = parse(testFiles, labelTransform)
    val (trainLabels, trainExamplesAndLabels) = parse(trainFiles, labelTransform)
    val model = modelTemplate.modelConstructor(numInputs, trainLabels.length)
    val id = modelTemplate.id

    // train
    trainExamplesAndLabels
      .toLocalIterator
      .grouped(batchSize)
      .map(batchToExamplesAndLabelsMatrix)
      .foreach { case (examples, labels) ⇒ model.fit(examples, labels) }

    // evaluate
    val batchModelEvaluations = testExamplesAndLabels
      .toLocalIterator
      .grouped(batchSize)
      .map(batchToExamplesAndLabelsMatrix)
      .map { case (examples, labels) ⇒ (id, ModelEvaluation(model, examples, labels ))}

    // fold evaluation results
    val modelEvaluations = batchModelEvaluations.foldLeft(Map[ModelTemplate.Id, ModelEvaluation]()) { case (result, (id, modelEvaluation)) ⇒
      result.updated(id, result.get(id).map(_ += modelEvaluation).getOrElse(modelEvaluation))
    }

    // save
    val Some(evaluation) = modelEvaluations.get(id)

    println(s"Model $id")
    println(evaluation.toPrettyString(testLabels))

    persistor.persist(id, model, testLabels, evaluation)
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
    val persistor = new ModelPersistor(outputPath)

    modelTemplates.foreach(x(sc.wholeTextFiles(trainPath), sc.wholeTextFiles(testPath), labelTransform, persistor))

//    val trainer = new ModelTrainer()
//    val train = dataSet(sc.wholeTextFiles(trainPath))(labelTransform)
//    val test  = dataSet(sc.wholeTextFiles(testPath))(labelTransform)
//    val best  = trainer.execute(model, train, test)
//    println(best)
  }
}
