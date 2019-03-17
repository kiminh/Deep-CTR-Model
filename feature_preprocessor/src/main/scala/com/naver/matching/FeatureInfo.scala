package com.naver.matching

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.collection.Map

trait FeatureInfo{
  def useDatabase : Unit

  def generateStringIndexer(fromDate: String, toDate: String) : Unit

  def loadStatisicalData(fromDate: String, toDate: String, bucketId: String): DataFrame

  def createExtraFeatures(dataDF: DataFrame,
    bcStatMap: Broadcast[Map[String, (Double, Double, Double)]],
    bcPositionBiasMap: Broadcast[Map[String, Double]],
    bcAdxChannelIdCTRMap: Broadcast[Map[String, Double]],
    bcAdxMediaUIIdCTRMap: Broadcast[Map[String, Double]]): DataFrame

  def fillNAValues(dataDF: DataFrame): DataFrame

  def normalizeNumericFeature(dataDF: DataFrame): DataFrame

  def transformedFeatures(dataDF: DataFrame): DataFrame

  def loadTrainData(fromDate: String, toDate: String,
    samplingRate: Double, negativeSamplingRate: Double,
    clickFractions: Map[String, Double], nonClickFractions: Map[String, Double]): DataFrame

  def loadInferData(date: String): DataFrame

  def saveAsTextfile(dataDF: DataFrame, dirName: String): Unit
}
