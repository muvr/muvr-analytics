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

// We don't actually use all these dependencies, but they are shown for the
// examples that explicitly use Hadoop.
object Dependency {
  private object versions {
    val spark        = "1.5.2"
    val scalaTest    = "2.2.4"
    val scalaCheck   = "1.12.2"
    val nd4j         = "0.4-rc3.6"
    val dl4j         = "0.4-rc3.6"
    val canova       = "0.0.0.12"
  }

  val sparkCore      = "org.apache.spark"  %% "spark-core"      % versions.spark
  val sparkStreaming = "org.apache.spark"  %% "spark-streaming" % versions.spark
  val sparkSQL       = "org.apache.spark"  %% "spark-sql"       % versions.spark
  val sparkHiveSQL   = "org.apache.spark"  %% "spark-hive"      % versions.spark
  val sparkRepl      = "org.apache.spark"  %% "spark-repl"      % versions.spark

  val dl4jCore       = "org.deeplearning4j" % "deeplearning4j-core" % versions.dl4j
  val nd4jX86        = "org.nd4j"           % "nd4j-x86"            % versions.nd4j
  val nd4jCuda       = "org.nd4j"           % "nd4j-jcublas-7.5"    % versions.nd4j

  val scalaTest      = "org.scalatest"     %% "scalatest"       % versions.scalaTest  % "test"
  val scalaCheck     = "org.scalacheck"    %% "scalacheck"      % versions.scalaCheck % "test"
}

object Dependencies {
  import Dependency._

  val em =
    Seq(sparkCore, sparkStreaming, sparkSQL, sparkHiveSQL, // sparkRepl,
      scalaTest, scalaCheck, dl4jCore, nd4jCuda)
}

object EmBuild extends Build {
  import Resolvers._
  import Dependencies._
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


