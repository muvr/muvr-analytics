package io.muvr.em

/**
  * Does what it says on the tin: performs very naïve argument parsing: the
  * arguments must appear in pairs: ``--name`` ``value`` as separate elements
  * in the ``params``.
  * @param params the parameters
  */
class NaiveArgumentParser(params: Array[String]) {

  private def simpleParseNamedParams(params: List[String]): List[(String, String)] = params match {
    case Nil ⇒ Nil
    case name::value::ps if name.startsWith("--") ⇒ (name.substring(2), value) :: simpleParseNamedParams(ps)
    case _::ps ⇒ simpleParseNamedParams(ps)
  }
  private val namedParams = simpleParseNamedParams(params.toList)

  /**
    * Gets the value of the parameter with the given ``name``
    * @param name the parameter name without the initial ``--``
    * @return the parameter value, if present
    */
  def get(name: String): Option[String] = namedParams.find(_._1 == name).map(_._2)

  /**
    * Gets the value of the parameter with the given ``name``, returning
    * ``default`` if missing
    *
    * @param name the parameter name without the initial ``--``
    * @param default the default value
    * @return the parameter value or default value
    */
  def getOrElse(name: String, default: String): String = get(name).getOrElse(default)

}
