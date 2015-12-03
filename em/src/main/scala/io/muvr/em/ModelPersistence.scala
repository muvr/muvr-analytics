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
  * @param s3Path the path in form ``s3n://``, ``bucket-name``, ``/prefix``
  */
class S3ModelPersistor(s3Path: String) extends ModelPersistor {

  /** Parses the ``s3Path`` into the appropriate elements */
  private val (awsAccessKey, awsSecretAccessKey, bucketName, bucketPrefix) = {
    val p = """s3n://(.*):(.*)@([^/]+)/?(.*)""".r
    s3Path match {
      case p(ak, ask, b, bp) ⇒ (ak, ask, b, bp)
     }
  }

  /** Credentials from the parsed elements */
  private lazy val credentials = new BasicAWSCredentials(awsAccessKey, awsSecretAccessKey)
  /** The S3 client */
  private lazy val client = new AmazonS3Client(credentials)

  /**
    * An implementation of the ``OutputStream``, which uploads the saved contents
    * on ``close()``.
    *
    * @param name the "file" name
    */
  private class AWSOutputStream(name: String) extends OutputStream {
    val tempFile = File.createTempFile(name, name)
    val fos = new FileOutputStream(tempFile)

    override def close(): Unit = {
      client.putObject(bucketName, s"$bucketPrefix/$name", tempFile)
      fos.close()
    }

    override def write(b: Int): Unit = fos.write(b)
  }

  type Handle = URL

  def getOutput(name: String): (OutputStream, Handle) = {
    val handle = new URL("https", "s3-eu-west-1.amazonaws.com", s"$bucketName/$bucketPrefix/$name")
    (new AWSOutputStream(name), handle)
  }

}

/**
  * Local file-based persistor
  * @param rootDirectory the base / root directory. If it does not exist, it will be created.
  */
class LocalFileModelPersistor(rootDirectory: String) extends ModelPersistor {
  /** Reference to the existing root directory */
  private val rootDirectoryFile = {
    val f = new File(rootDirectory)
    f.mkdirs()
    require(f.exists(), s"The output directory $f does not exist and could not be created.")
    f
  }

  type Handle = File

  def getOutput(name: String): (OutputStream, Handle) = {
    val file = new File(rootDirectoryFile, name)
    (new FileOutputStream(file), file)
  }

}

/**
  * Companion object for the persistor; given an instance of the trait, constructs a
  * ``ModelPersistor.Type``, which is a function that performs the required save
  * operation.
  */
object ModelPersistor {

  /**
    * The save function for the given handle type
    * @tparam Handle the handle type
    */
  type Type[Handle] = TrainedAndEvaluatedModel ⇒ PersistedModel[Handle]

  /**
    * Given an instance of ``ModelPersistor``, return the ``ModelPersistor.Type[Handle]``, where
    * ``Handle`` is the ``Handle`` member of the ``modelPersistor`` instance.
    *
    * @param modelPersistor the model persistor instance
    * @param tem the TEM
    * @return handles to all saved elements of ``tem``
    */
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
