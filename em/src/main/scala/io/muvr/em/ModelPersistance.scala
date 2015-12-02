package io.muvr.em

import java.io._
import java.net.URL

import org.nd4j.linalg.factory.Nd4j

case class PersistedModel[Handle](configuration: Handle, params: Handle, labels: Handle, evaluation: Handle, confusionMatrix: Handle)

trait ModelPersistor {
  type Handle
  def getOutputStream(name: String): (OutputStream, Handle)
}

class S3ModelPersistor(bucket: String) extends ModelPersistor {
  type Handle = URL

  def getOutputStream(name: String): (OutputStream, Handle) = ???

}

class LocalFileModelPersistor(rootDirectory: String) extends ModelPersistor {
  private val rootDirectoryFile = {
    val f = new File(rootDirectory)
    f.mkdirs()
    f
  }

  type Handle = File

  def getOutputStream(name: String): (OutputStream, Handle) = {
    val file = new File(rootDirectoryFile, name)
    (new FileOutputStream(file), file)
  }

}

object ModelPersistor {

  def apply(modelPersistor: ModelPersistor)(tem: TrainedAndEvaluatedModel): PersistedModel[modelPersistor.Handle] = {
    val (paramsOut, paramsA) = modelPersistor.getOutputStream(s"${tem.id}-params.raw")
    Nd4j.write(tem.model.params(), new DataOutputStream(paramsOut))

    val (configurationOut, configurationA) = modelPersistor.getOutputStream(s"${tem.id}-configuration.json")
    configurationOut.write(tem.model.conf().toJson.getBytes("UTF-8"))
    configurationOut.close()

    val (labelsOut, labelsA) = modelPersistor.getOutputStream(s"${tem.id}-labels.txt")
    labelsOut.write(tem.labels.labels.mkString("\n").getBytes("UTF-8"))
    labelsOut.close()

    val (confusionMatrixOut, confusionMatrixA) = modelPersistor.getOutputStream(s"${tem.id}-cm.csv")
    tem.evaluation.saveConfusionMatrixAsCSV(tem.labels, new BufferedWriter(new OutputStreamWriter(confusionMatrixOut)))
    confusionMatrixOut.close()

    val (evaluationOut, evaluationA) = modelPersistor.getOutputStream(s"${tem.id}-evaluation.csv")
    tem.evaluation.saveEvaluationAsCSV(new BufferedWriter(new OutputStreamWriter(evaluationOut)))
    evaluationOut.close()

    PersistedModel(configurationA, paramsA, labelsA, evaluationA, confusionMatrixA)
  }

}
