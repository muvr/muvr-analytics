package io.muvr.em

import java.io._

import io.muvr.em.dataset.Labels
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

case class PersistedModel[Handle](configuration: Handle, params: Handle, labels: Handle, evaluation: Handle, confusionMatrix: Handle)

class ModelPersistor(rootDirectory: File) {

  def getOutputStream(name: String): (OutputStream, File) = {
    val file = new File(rootDirectory, name)
    (new FileOutputStream(file), file)
  }

  def persist(id: ModelTemplate.Id, model: MultiLayerNetwork, labels: Labels, modelEvaluation: ModelEvaluation): PersistedModel[File] = {
    val (paramsOut, paramsA) = getOutputStream(s"$id-params.raw")
    Nd4j.write(model.params(), new DataOutputStream(paramsOut))

    val (configurationOut, configurationA) = getOutputStream(s"$id-configuration.json")
    configurationOut.write(model.conf().toJson.getBytes("UTF-8"))
    configurationOut.close()

    val (labelsOut, labelsA) = getOutputStream(s"$id-labels.txt")
    labelsOut.write(labels.labels.mkString("\n").getBytes("UTF-8"))
    labelsOut.close()

    val (confusionMatrixOut, confusionMatrixA) = getOutputStream(s"$id-cm.csv")
    modelEvaluation.saveConfusionMatrixAsCSV(labels, new BufferedWriter(new OutputStreamWriter(confusionMatrixOut)))
    confusionMatrixOut.close()

    val (evaluationOut, evaluationA) = getOutputStream(s"$id-evaluation.csv")
    modelEvaluation.saveEvaluationAsCSV(new BufferedWriter(new OutputStreamWriter(evaluationOut)))
    evaluationOut.close()

    PersistedModel(configurationA, paramsA, labelsA, evaluationA, confusionMatrixA)
  }

}
