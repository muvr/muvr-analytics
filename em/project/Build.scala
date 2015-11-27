import sbt._
import sbt.Keys._

object BuildSettings {

  lazy val buildSettings = Defaults.coreDefaultSettings ++ Seq (
    name          := "em",
    version       := "1.0.0-SNAPSHOT",
    scalaVersion  := "2.11.7",
    organization  := "io.muvr",
    description   := "muvr exercise model",
    scalacOptions := Seq("-deprecation", "-unchecked", "-encoding", "utf8", "-Xlint")
  )
}


object Resolvers {
  val typesafe = "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"
  val sonatype = "Sonatype Release" at "https://oss.sonatype.org/content/repositories/releases"
  val mvnrepository = "MVN Repo" at "http://mvnrepository.com/artifact"

  val allResolvers = Seq(typesafe, sonatype, mvnrepository)

}

object Dependency {

  object spark {
    private val core = "org.apache.spark" %% "spark-core" % "1.5.2"

    val all = Seq(core)
  }

  object dl4j {
    private val nd4jVersion         = "0.4-rc3.6"
    private val dl4jVersion         = "0.4-rc3.6"

    private val dl4jCore            = "org.deeplearning4j"                % "deeplearning4j-core"     % dl4jVersion intransitive()
    private val nd4jX86             = "org.nd4j"                          % "nd4j-x86"                % nd4jVersion intransitive()
    private val jacksonDF           = "com.fasterxml.jackson.dataformat"  % "jackson-dataformat-yaml" % "2.5.4" intransitive()
    private val nd4jApi             = "org.nd4j"                          % "nd4j-api"                % nd4jVersion intransitive()
    private val nd4jBytebuddy       = "org.nd4j"                          % "nd4j-bytebuddy"          % nd4jVersion intransitive()
    private val bytebuddy           = "net.bytebuddy"                     % "byte-buddy"              % "0.7.2" intransitive()
    private val jblas               = "org.jblas"                         % "jblas"                   % "1.2.4" intransitive()
    private val springFrameworkCore = "org.springframework"               % "spring-core"             % "3.2.5.RELEASE" intransitive()
    private val netlib              = "com.github.fommil.netlib"          % "all"                     % "1.1.2" pomOnly()
    private val reflections         = "org.reflections"                   % "reflections"             % "0.9.9-RC1" intransitive()
    private val javassist           = "org.javassist"                     % "javassist"               % "3.19.0-GA" intransitive()

    val all = Seq(dl4jCore, nd4jApi, nd4jX86, jblas, netlib, nd4jBytebuddy, reflections, javassist, springFrameworkCore, jacksonDF, bytebuddy)
  }

}

object Dependencies {
  import Dependency._

  val em = spark.all ++ dl4j.all
}

object EmBuild extends Build {
  import Resolvers._
  import BuildSettings._

  val excludeSigFilesRE = """META-INF/.*\.(SF|DSA|RSA)""".r
  lazy val em = Project(
    id = "em",
    base = file("."),
    settings = buildSettings ++ Seq(
      shellPrompt := { state => "(%s)> ".format(Project.extract(state).currentProject.id) },
      maxErrors          := 5,
      triggeredMessage   := Watched.clearWhenTriggered,
      // runScriptSetting,
      resolvers := allResolvers,
      exportJars := true,
      // For the Hadoop variants to work, we must rebuild the package before
      // running, so we make it a dependency of run.
      (run in Compile) <<= (run in Compile) dependsOn (packageBin in Compile),
      libraryDependencies ++= Dependencies.em,
      excludeFilter in unmanagedSources := (HiddenFileFilter || "*-script.scala"),
      unmanagedResourceDirectories in Compile += baseDirectory.value / "conf",
      unmanagedResourceDirectories in Test += baseDirectory.value / "conf",
      mainClass := Some("run"),
      //This is important for some programs to read input from stdin
      connectInput in run := true,
      // Works better to run the examples and tests in separate JVMs.
      fork := true,
      // Must run Spark tests sequentially because they compete for port 4040!
      parallelExecution in Test := false))
}


