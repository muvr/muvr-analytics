package io.muvr.em

import java.io._
import java.net.URL

import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3Client
import org.nd4j.linalg.factory.Nd4j

/**
  * The persisted model detail carrying the handles to all resources that were produced
  *
  * @param configuration the model configuration
  * @param params the model parameters (weights, biases, ...)
  * @param labels the labels
  * @param evaluation the evaluation (accuracy, precision, ...)
  * @param confusionMatrix the confusion matrix
  * @tparam Handle the handle to find the resource (e.g. ``File`` or ``URL``)
  */
case class PersistedModel[Handle](configuration: Handle, params: Handle, labels: Handle, evaluation: Handle, confusionMatrix: Handle)

/**
  * Defines the persistence mechanism
  */
trait ModelPersistor {
  /**
    * The type this persistor produces
    */
  type Handle

  /**
    * Gets the output given its ``name``. The implementations _must_ call ``close()`` on the returned ``OutputStream``.
    *
    * @param name the name of the resource to create
    * @return the pair of ``OutputStream`` that can receive the bytes for the resource and the handle for it
    */
  def getOutput(name: String): (OutputStream, Handle)
}

/**
  * S3 target
  * @param bucket the bucket name
  */
class S3ModelPersistor(bucket: String, awsAccessKey: String, awsSecretAccessKey: String) extends ModelPersistor {
  private lazy val credentials = new BasicAWSCredentials(awsAccessKey, awsSecretAccessKey)
  private lazy val client = new AmazonS3Client(credentials)

  private class AWSOutputStream(name: String) extends OutputStream {
    val tempFile = File.createTempFile(name, name)
    val fos = new FileOutputStream(tempFile)

    override def close(): Unit = {
      client.putObject(bucket, name, tempFile)
      fos.close()
    }

    override def write(b: Int): Unit = fos.write(b)
  }

  type Handle = URL

  def getOutput(name: String): (OutputStream, Handle) = {
    val handle = new URL("s3", "s3.amazonmagic.com", "foo")
    (new AWSOutputStream(name), handle)
  }

}

/**
  * Local file-based persistor
  * @param rootDirectory the base / root directory. If it does not exist, it will be created.
  */
class LocalFileModelPersistor(rootDirectory: String) extends ModelPersistor {
  private val rootDirectoryFile = {
    val f = new File(rootDirectory)
    f.mkdirs()
    f
  }

  type Handle = File

  def getOutput(name: String): (OutputStream, Handle) = {
    val file = new File(rootDirectoryFile, name)
    (new FileOutputStream(file), file)
  }

}

object ModelPersistor {

  type Type[Handle] = TrainedAndEvaluatedModel ⇒ PersistedModel[Handle]

  def apply(modelPersistor: ModelPersistor)(tem: TrainedAndEvaluatedModel): PersistedModel[modelPersistor.Handle] = {
    val (paramsOut, paramsA) = modelPersistor.getOutput(s"${tem.id}-params.raw")
    Nd4j.write(tem.model.params(), new DataOutputStream(paramsOut))

    val (configurationOut, configurationA) = modelPersistor.getOutput(s"${tem.id}-configuration.json")
    configurationOut.write(tem.model.conf().toJson.getBytes("UTF-8"))
    configurationOut.close()

    val (labelsOut, labelsA) = modelPersistor.getOutput(s"${tem.id}-labels.txt")
    labelsOut.write(tem.labels.labels.mkString("\n").getBytes("UTF-8"))
    labelsOut.close()

    val (confusionMatrixOut, confusionMatrixA) = modelPersistor.getOutput(s"${tem.id}-cm.csv")
    tem.evaluation.saveConfusionMatrixAsCSV(tem.labels, new BufferedWriter(new OutputStreamWriter(confusionMatrixOut)))
    confusionMatrixOut.close()

    val (evaluationOut, evaluationA) = modelPersistor.getOutput(s"${tem.id}-evaluation.csv")
    tem.evaluation.saveEvaluationAsCSV(new BufferedWriter(new OutputStreamWriter(evaluationOut)))
    evaluationOut.close()

    PersistedModel(configurationA, paramsA, labelsA, evaluationA, confusionMatrixA)
  }

}
