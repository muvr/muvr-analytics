package io.muvr.em

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
    * @param persistor the function that will be used to persist
    * @param modelTemplate the model template
    */
  private def pipeline[A](train: LabelsAndEL, test: LabelsAndEL, persistor: ModelPersistor.Type[A])(modelTemplate: ModelTemplate): PersistedModel[A] = {
    val batchSize = 50000
    val (testLabels, testExamplesAndLabels) = test
    val (trainLabels, trainExamplesAndLabels) = train
    val initialModel = modelTemplate.modelConstructor(numInputs, trainLabels.length)
    val id = modelTemplate.id

    // train
    val trainedModel = trainExamplesAndLabels
      .coalesce(1)
      .mapPartitions(_.grouped(batchSize).map(batchToExamplesAndLabelsMatrix).map((initialModel, _)))
      .map { case (model, (examples, labels)) ⇒ model.fit(examples, labels); model }
      .take(1)
      .head


    // evaluate
    val evaluation = testExamplesAndLabels
      .mapPartitions(_.grouped(batchSize).map(batchToExamplesAndLabelsMatrix))
      .map { case (examples, labels) ⇒ ModelEvaluation(trainedModel, examples, labels ) }
      .reduce(_ + _)

    // eyeball result
    println(s"Model $id")
    println(evaluation.toPrettyString(testLabels))

    // save
    persistor(TrainedAndEvaluatedModel(id, trainedModel, trainLabels, evaluation))
  }

  /**
    * Returns the label transform function depending on whether we're building exercise classifier
    * or exercise vs. slacking classifier
    *
    * @param slacking true to build exercise classifier; false to build E vs. S classifier
    * @return the label transform
    */
  private def buildLabelTransform(slacking: Boolean): LabelTransform = {
    if (slacking) {
      case "" ⇒ Some("-")
      case _ ⇒ Some("e")
    } else {
      case "" ⇒ None
      case "triceps-dips" ⇒ None
      case "dumbbell-bench-press" ⇒ None
      case x ⇒ Some(x)
    }
  }

  /**
    * Starts the Spark app
    * @param args the args
    */
  def main(args: Array[String]): Unit = {
    val (s, h) = new S3ModelPersistor("s3n://AKIAIE55HIA7FQHRCJLQ:4c9DK1g1GNP78YzLRun048GiWOUn+8kxSpMw3fA7@muvr-open-training-data/models").getOutput("foo")
    s.write(65)
    s.close()
    println(h)
    return

    val parser = new NaiveArgumentParser(args)

    // parse required arguments
    val master           = Option(System.getenv("spark.master")).getOrElse("local")
    val Some(model)      = parser.get("model")
    val Some(trainPath)  = parser.get("train-path")
    val Some(testPath)   = parser.get("test-path")
    val Some(outputPath) = parser.get("output-path")
    val labelTransform   = buildLabelTransform(parser.getOrElse("slacking", "") == "true")
    val persistor        = if (outputPath.startsWith("s3n://")) new S3ModelPersistor(outputPath) else new LocalFileModelPersistor(outputPath)

    // construct the Spark Context, run the training pipeline
    val name = "ModelTrainer"
    val conf = new SparkConf().
      setMaster(master).
      setAppName(name).
      set("spark.app.id", name)
    val sc = new SparkContext(conf)
    sc.hadoopConfiguration.get("")

    val train = parse(sc.wholeTextFiles(s"$trainPath/$model"), labelTransform)
    val test  = parse(sc.wholeTextFiles(s"$testPath/$model"),  labelTransform)

    val evaluatedModels = modelTemplates.map(pipeline(train, test, ModelPersistor(persistor)))
    evaluatedModels.foreach(println)
  }

}
