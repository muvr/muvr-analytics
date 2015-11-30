#!/bin/bash

# SPARK_HOME=/usr/local/Cellar/apache-spark/1.5.2

sbt clean assembly

#SPARK_HOME/bin
spark-submit --class io.muvr.em.ModelTrainerMain --driver-memory 4G target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar 
