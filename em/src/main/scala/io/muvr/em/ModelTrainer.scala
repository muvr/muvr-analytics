package io.muvr.em

import java.io.File

import io.muvr.em.dataset.ExerciseDataSet.DataSet
import io.muvr.em.dataset.{ExerciseDataSet, Labels}
import io.muvr.em.model.MLP
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

class ModelTrainer(persistor: ModelPersistor) {


  /** The various model templates */
  private val modelTemplates: List[ModelTemplate] = List.fill(10)(MLP.shallowModel)

  /**
    * Train the model created by ``modelConstructor`` on the given ``dataSet``, with some specific ``modelName``
    * @param dataSet the training data set
    * @param modelConstructor the model constructor
    * @return the MLN
    */
  private def train(dataSet: DataSet, modelConstructor: ModelTemplate.Constructor): MultiLayerNetwork = {
    // construct the "empty" model
    val model = modelConstructor(dataSet.numInputs, dataSet.numOutputs)
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    // fit the examples and labels in the model
    model.fit(examplesMatrix, labelsMatrix)
    model
  }

  /**
    * Saves the model configuration and parameters, and labels
    *
    * @param modelName the model name
    * @param model the model
    * @param labels the labels
    */
  private def save(modelName: String, model: MultiLayerNetwork, modelEvaluation: ModelEvaluation, labels: Labels): PersistedModel[File] =
    persistor.persist(modelName, model, labels, modelEvaluation)


  /**
    * Evaluates the given ``model`` on the test ``dataSet`` with human-readable ``labels``
    * @param model the (trained) model
    * @param labels the label strings
    * @param dataSet the test data set
    * @return the accuracy
    */
  private def evaluate(model: MultiLayerNetwork, labels: Labels, dataSet: DataSet): ModelEvaluation = {
    val (examplesMatrix, labelsMatrix) = dataSet.examples
    ModelEvalutaion(model, examplesMatrix, labelsMatrix)
  }

  /**
    * Model pipeline is to construct a model, train on the train data set, evaluate on the test data set, save
    * and return model identity and accuracy
    *
    * @param trainDataSet the train data set
    * @param testDataSet the test data set
    * @param model the model metadata
    * @return model id and accuracy
    */
  private def pipeline(tag: String, trainDataSet: DataSet, testDataSet: DataSet)(model: ModelTemplate): (PersistedModel[File], Double) = {
    val m = train(trainDataSet, model.modelConstructor)
    val cm = evaluate(m, testDataSet.labels, testDataSet)
    val pm = save(s"$tag-${model.id}", m, cm, trainDataSet.labels)

    println(model.id)
    println(cm.toPrettyString(testDataSet.labels))
    println()

    (pm, 5 * cm.accuracy() + cm.f1() + cm.precision() + cm.recall())
  }

  def execute(tag: String, train: ExerciseDataSet.DataSet, test: ExerciseDataSet.DataSet): (PersistedModel[File], Double) ={
    // train, evaluate, and save each model on the data, selecting the best one
    modelTemplates.map(pipeline(tag, train, test)).maxBy(_._2)
  }

}
