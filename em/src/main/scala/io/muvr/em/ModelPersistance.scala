package io.muvr.em

import java.io._

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object ModelPersistance {

  final implicit class MultiLayerNetworkPersistence(model: MultiLayerNetwork) {

    def save(params: OutputStream, conf: OutputStream): Unit = {
      Nd4j.write(model.params(), new DataOutputStream(params))
      conf.write(model.conf().toYaml.getBytes("UTF-8"))
      conf.close()
    }

    def save(rootDirectory: String, name: String): Unit =
      save(new FileOutputStream(s"$rootDirectory/models/$name.params"),
           new FileOutputStream(s"$rootDirectory/models/$name.conf"))

  }

  def load(params: InputStream, conf: InputStream): MultiLayerNetwork = {
    val loadedParams = Nd4j.read(params)
    val loadedConf = Source.fromInputStream(conf, "UTF-8").mkString
    new MultiLayerNetwork(loadedConf, loadedParams)
  }

  def load(rootDirectory: String, name: String): MultiLayerNetwork =
    load(new FileInputStream(s"$rootDirectory/models/$name.params"),
         new FileInputStream(s"$rootDirectory/models/$name.conf"))

}
