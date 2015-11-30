package io.muvr.em

class ArgumentParser(params: Array[String]) {

  private def simpleParseNamedParams(params: List[String]): List[(String, String)] = params match {
    case Nil ⇒ Nil
    case name::value::ps if name.startsWith("--") ⇒ (name.substring(2), value) :: simpleParseNamedParams(ps)
    case _::ps ⇒ simpleParseNamedParams(ps)
  }
  private val namedParams = simpleParseNamedParams(params.toList)

  def get(name: String): Option[String] = namedParams.find(_._1 == name).map(_._2)
  def getOrElse(name: String, default: String): String = get(name).getOrElse(default)

}
