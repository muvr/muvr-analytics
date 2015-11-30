package io.muvr.em

import java.io.File

import io.muvr.em.dataset.{ExerciseDataSetFile, Labels}
import io.muvr.em.model.MLP
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object ModelTrainerMain {
  private val numInputs = 1200
  private val modelTemplates: List[ModelTemplate] = List.fill(5)(MLP.shallowModel)

  private type LabelsAndEL = (Labels, RDD[(INDArray, INDArray)])
  private type LabelTransform = String ⇒ Option[String]

  /**
    * Parses the contents of CSV files into a collected list of labels and RDDs of examples and labels
    * @param files the RDD with file contents
    * @param labelTransform the label transformation to apply
    * @return distinct labels and RDD of (examples, labels)
    */
  private def parse(files: RDD[(String, String)], labelTransform: LabelTransform): LabelsAndEL = {
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

  /**
    * Converts sequence of a vector of examples and vector of labels into matrix of examples & labels
    * @param batch the batch of example and label vectors
    * @return the matrix of examples and labels
    */
  private def batchToExamplesAndLabelsMatrix(batch: Seq[(INDArray, INDArray)]): (INDArray, INDArray) = {
    val (_, labels) = batch.head
    val examplesMatrix = Nd4j.create(batch.length, numInputs)
    val labelsMatrix = Nd4j.create(batch.length, labels.length())
    batch.zipWithIndex.foreach { case ((example, label), i) ⇒
      examplesMatrix.putRow(i, example)
      labelsMatrix.putRow(i, label)
    }
    (examplesMatrix, labelsMatrix)
  }

  /**
    * The learning pipeline constructs the model from the ``modelTemplate``, supplying the
    * right number of inputs and outputs; then fits the model on all training data, then
    * evaluates the model's performance on the test data. Finally, it persist the model
    * and evaluation.
    *
    * @param train the names and content of the training files (CSVs)
    * @param test the names and content of the test files (CSVs)
    * @param outputPath the directory to write the outputs to
    * @param modelTemplate the model template
    */
  private def pipeline(train: LabelsAndEL, test: LabelsAndEL, outputPath: File)(modelTemplate: ModelTemplate): Unit = {
    val batchSize = 50000
    val (testLabels, testExamplesAndLabels) = test
    val (trainLabels, trainExamplesAndLabels) = train
    val model = modelTemplate.modelConstructor(numInputs, trainLabels.length)
    val id = modelTemplate.id

    // train
    trainExamplesAndLabels
//      .mapPartitions(_.grouped(batchSize).map(batchToExamplesAndLabelsMatrix))
      .toLocalIterator.grouped(batchSize).map(batchToExamplesAndLabelsMatrix)
      .foreach { case (examples, labels) ⇒ model.fit(examples, labels) }

    // evaluate
    testExamplesAndLabels
      .mapPartitions(_.grouped(batchSize).map(batchToExamplesAndLabelsMatrix))
//      .toLocalIterator.grouped(batchSize).map(batchToExamplesAndLabelsMatrix)
      .map { case (examples, labels) ⇒ ModelEvaluation(model, examples, labels ) }
      .foreach { evaluation ⇒
        // save
        ModelPersistor.persist(outputPath, id, model, testLabels, evaluation)

        // eyeball result
        println(s"Model $id")
        println(evaluation.toPrettyString(testLabels))
      }
  }

  /**
    * Returns the label transform function depending on whether we're building exercise classifier
    * or exercise vs. slacking classifier
    *
    * @param exercising true to build exercise classifier; false to build E vs. S classifier
    * @return the label transform
    */
  private def labelTransform(exercising: Boolean): LabelTransform = {
    if (exercising) {
      case "" ⇒ None
      case "triceps-dips" ⇒ None
      case "dumbbell-bench-press" ⇒ None
      case x ⇒ Some(x)
    } else {
      case "" ⇒ Some("-")
      case _ ⇒ Some("e")
    }
  }

  /**
    * Starts the Spark app
    * @param args the args
    */
  def main(args: Array[String]): Unit = {
    val parser = new ArgumentParser(args)

    // parse required arguments
    val Some(master)     = parser.get("master")
    val Some(model)      = parser.get("model")
    val Some(trainPath)  = parser.get("train-path")
    val Some(testPath)   = parser.get("test-path")
    val Some(outputPath) = parser.get("output-path")
    val labelTransform   = labelTransform(parser.getOrElse("exercising", "true") == "true")
    val outputPathFile   = new File(outputPath); outputPathFile.mkdirs()

    // construct the Spark Context, run the training pipeline
    val name = "ModelTrainer"
    val conf = new SparkConf().
      setMaster(master).
      setAppName(name).
      set("spark.app.id", name)
    val sc = new SparkContext(conf)

    val train = parse(sc.wholeTextFiles(s"$trainPath/$model"), labelTransform)
    val test  = parse(sc.wholeTextFiles(s"$testPath/$model"),  labelTransform)
    modelTemplates.foreach(pipeline(train, test, outputPathFile))
  }
}
