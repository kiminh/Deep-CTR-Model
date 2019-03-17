package com.naver.matching

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.functions.{min, max}
import org.apache.spark.sql.hive.HiveContext
//import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer

import org.apache.log4j.{Level, Logger, LogManager}

import scala.collection.Map

object FeaturePreprocessor{
  val log = LogManager.getRootLogger

  val ssaMetaTableName = "t_meta_5"
  val ssaFeatureTableName = "v_features"
  val ssaDatabaseName = "ssa"

  val saMetaTableName = "t_meta_3"
  val saFeatureTableName = "t_features_3"
  val saDatabaseName = "sa"

  def main(args: Array[String]): Unit = {

    val parser = new scopt.OptionParser[CommandLineArgs]("scopt") {
      head("scopt", "3.x")

      opt[String]("ad_type").action{ (x, c) =>
        c.copy(adType = x)}.text("advertising type")

      opt[String]("data_type").action{ (x, c) =>
        c.copy(dataType = x)}.text("data type")

      opt[String]("save_dir").required.action{ (x, c) =>
        c.copy(saveDir = x)}.text("data type")
     
      opt[Double]("ns_rate").action{ (x, c) =>
        c.copy(negativeSamplingRate = x)}.text("negative sampling rate")

      opt[Double]("sampling_rate").action{ (x, c) =>
        c.copy(samplingRate = x)}.text("sampling rate")

      opt[String]("from_date").action{ (x, c) =>
        c.copy(fromDate = x)}.text("from date")

      opt[String]("to_date").action{ (x, c) =>
        c.copy(toDate = x)}.text("to date")

      opt[String]("infer_date").action{ (x, c) =>
        c.copy(inferDate = x)}.text("infer date")

      opt[String]("str_to_int_dir").required.action{ (x, c) =>
        c.copy(stringToIntDir = x)}.text("to date")

      opt[Int]("period").action{ (x, c) =>
        c.copy(period = x)}.text("period")

      opt[String]("bucket_id").action{ (x, c) =>
        c.copy(bucketId = x)}.text("bucket id")

      opt[Boolean]("gen_string_indexer").action{ (_, c) =>
        c.copy(genStringIndexer = true)}.text("generate string_indexer model")
    }

    parser.parse(args, CommandLineArgs()) match {
      case Some(config) =>
        implicit val spark: SparkSession = SparkSession
          .builder()
          .appName("preprocess_feature")
          .config("hive.metastore.uris", 
              "thrift://amatchhdp001-sa.nfra.io:9083,thrift://amatchhdp005-sa.nfra.io:9083")
          .config("mapreduce.fileoutputcommitter.algorithm.version", "2")
          //.config("spark.sql.warehouse.dir", "/user/hive/warehouse")
          //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
          .enableHiveSupport()
          .config("hive.exec.dynamic.partition", "true")
          .config("hive.exec.dynamic.partition.mode", "nonstrict")
          .getOrCreate()

        try {
          val toDate = config.getToDate
          val fromDate = config.getFromDate
          log.info(s"period ${config.period}")
          log.info(s"infer date ${config.getInferDate}")
          log.info(s"fp starts with ($fromDate , $toDate)")
          
	  run(config.adType, config.dataType, config.saveDir,
            config.samplingRate, config.negativeSamplingRate, 
            config.getFromDate, config.getToDate,
            config.getStatFromDate, config.getStatToDate,
            config.stringToIntDir, config.genStringIndexer,
            config.getInferDate, config.bucketId)
        } finally{
          spark.close
        }
      case None =>
	println("parsing a command line errors")

    }
  }
	
  def run(adType: String, dataType: String, saveDir: String,
     samplingRate: Double, negativeSamplingRate: Double, fromDate: String,
     toDate: String, statFromDate: String, statToDate: String,
     stringToIntDir: String, genStringIndexer: Boolean, 
     inferDate: String, bucketId: String)(implicit spark: SparkSession) = {

    import spark.implicits._
    
    val featureInfo = 
      adType match {
        case "ssa" =>
          SSAFeatureInfo(dataType, ssaDatabaseName, ssaFeatureTableName,
	    ssaMetaTableName, stringToIntDir, genStringIndexer)
	case "sa" =>
          SAFeatureInfo(dataType, saDatabaseName, saFeatureTableName,
            saMetaTableName, stringToIntDir, genStringIndexer)
	case _ =>
          SSAFeatureInfo(dataType, ssaDatabaseName, ssaFeatureTableName,
            ssaMetaTableName, stringToIntDir, genStringIndexer)
    }

    // use database
    featureInfo.useDatabase

    // generate StringIndexer and save it
    featureInfo.generateStringIndexer(fromDate, toDate)

    //get ad stat info
    val statDataDF : DataFrame = featureInfo.loadStatisicalData(statFromDate, statToDate, bucketId)
   	
    // caching
    statDataDF.cache()
  
    val (statMap, positionBiasMap, adxChannelIdCTRMap, adxMediaUIIdCTRMap) = 
      _getAllBcStatisicalMap(adType, statDataDF)
  
    // uncaching
    statDataDF.unpersist()

    val dataDF = 
     if (dataType == "train" || dataType == "eval") { 
        val nonClickFractions : Map[String, Double] = statMap.map(x => (x._1, samplingRate*negativeSamplingRate)) 
        val clickFractions : Map[String, Double] = statMap.map(x => (x._1, samplingRate)) 
        featureInfo.loadTrainData(fromDate, toDate, samplingRate,
          negativeSamplingRate, clickFractions, nonClickFractions)
      } else { // infer
        //val metaDate = _currentMetaDate 
        featureInfo.loadInferData(inferDate)
      }
    
    //dataDF.cache()

    //Replace NA with default 
    val _dataDF = featureInfo.fillNAValues(dataDF)  

    //broadcast variables
    val sc = spark.sparkContext
    val bcStatMap = sc.broadcast(statMap)
    val bcPositionBiasMap = sc.broadcast(positionBiasMap)
    val bcAdxChannelIdCTRMap = sc.broadcast(adxChannelIdCTRMap)
    val bcAdxMediaUIIdCTRMap = sc.broadcast(adxMediaUIIdCTRMap)

    //Add Extra Column to Feature
    val dataDFExtra = featureInfo.createExtraFeatures(
      _dataDF, bcStatMap, bcPositionBiasMap, bcAdxChannelIdCTRMap, bcAdxMediaUIIdCTRMap)

    // Normalized numeric features
    val dataDFNormalized = featureInfo.normalizeNumericFeature(dataDFExtra)

    //Transform feature 
    val allDataDF = featureInfo.transformedFeatures(dataDFNormalized)
 
    //Save dataframe as table
    featureInfo.saveAsTextfile(allDataDF, saveDir)    
  }

  /*
  def _currentMetaDate : String = {
    val dateFormat = new SimpleDateFormat("yyyyMMdd")
    val cal = Calendar.getInstance()
    val hourOfDay = cal.get(Calendar.HOUR_OF_DAY)
    if (hourOfDay < 1) {
      cal.add(Calendar.DATE, -1)
      dateFormat.format(cal.getTime)
    } else {
      dateFormat.format(cal.getTime)
    }
  }*/

  def _getIdImpressionAndClickMap(idName:String, dataDF: DataFrame)(implicit spark: SparkSession) = {
    import spark.implicits._

    dataDF
      .select(idName, "clk", "imp")
      .groupBy(idName)
      .sum("clk", "imp")
      .map{
        case Row(id: String, click:Long, impression: Long) =>
          (id, (click.toDouble, impression.toDouble))}
      .collect()
      .toMap
  }

  def _getPairIdImpressionAndClipMap(idName1: String, idName2: String, dataDF: DataFrame)(implicit spark: SparkSession) = {
    import spark.implicits._
    dataDF
      .select(idName1, idName2, "clk", "imp")
      .groupBy(idName1, idName2)
      .sum("clk", "imp")
      .map{
        case Row(id1: String, id2: String, click: Long, impression: Long) =>
          (id1+id2, (click.toDouble, impression.toDouble))
      }
      .collect()
      .toMap
  }

  def _getPairIdCTRMap(idName: String, idName2: String, dataDF: DataFrame)(implicit spark: SparkSession) = {
    import spark.implicits._

    dataDF
      .select(idName, idName2, "clk", "imp")
      .groupBy(idName, idName2)
      .sum("clk", "imp")
      .map{
        case Row(id: String, id2: String, click: Long, impression: Long) =>
          (id+id2, click.toDouble/impression.toDouble)
      }
      .collect()
      .toMap
  }

  def _getIdCTRMap(idName: String, dataDF: DataFrame)(implicit spark: SparkSession) = {
    import spark.implicits._

    dataDF
      .select(idName, "clk", "imp")
      .groupBy(idName)
      .sum("clk", "imp")
      .map{
        case Row(id: String, click: Long, impression: Long) =>
          (id, click.toDouble/impression.toDouble)
      }
      .collect()
      .toMap
  }

  // [Ad -> (COEC, Click, Impression)]
  def _getAdStatisicalMap(dataDF: DataFrame, adId: String, bcPositionBiasMap: Broadcast[Map[String, Double]],
    bcAdImpAndClickMap: Broadcast[Map[String, (Double, Double)]]) (implicit spark: SparkSession) : Map[String, (Double, Double, Double)]= {
    import spark.implicits._ 

    dataDF
      .select(adId, "total_rank", "imp")
      .groupBy(adId, "total_rank")
      .sum("imp")
      .map {
        case Row(adId: String, position: String, impression: Long) =>
          val positionCTR = bcPositionBiasMap.value(position)
          (adId, impression*positionCTR)
      }.groupByKey(_._1)
      .reduceGroups{ (a, b) =>
        (a._1, a._2 + b._2) }
      .map(_._2)
      .map{ case (adKey, impressionPosCTR) =>
        val (click, impression) = bcAdImpAndClickMap.value(adKey)
        (adKey, (click / impressionPosCTR, click, impression))}
      .collect()
      .toMap
  }

  //[AdxId -> (COEC, Click, Impression)]
  def _getAdxIdStatisicalMap(dataDF: DataFrame, adId: String, idName: String,
    bcPositionBiasMap: Broadcast[Map[String, Double]],
    bcAdxIdImpAndClickMap: Broadcast[Map[String, (Double, Double)]])(implicit spark: SparkSession) = {
    import spark.implicits._

    dataDF
      .select(adId, idName, "total_rank", "imp")
      .groupBy(adId, idName, "total_rank")
      .sum("imp")
      .map {
        case Row(adId: String, id: String, position: String, impression: Long) =>
          val positionCTR = bcPositionBiasMap.value(position)
          (adId+id, impression*positionCTR)
      }.groupByKey(_._1)
      .reduceGroups{ (a, b) =>
        (a._1, a._2 + b._2) }
      .map(_._2)
      .map{ case (adKey, impressionPosCTR) =>
        val (click, impression) = bcAdxIdImpAndClickMap.value(adKey)
        (adKey, (click / impressionPosCTR, click, impression))}
      .collect()
      .toMap
  }

  def _getAllBcStatisicalMap(adType: String, statDataDF: DataFrame)(implicit spark: SparkSession) = {
    val sc = spark.sparkContext

    import spark.implicits._

    // ad click, impression Map
    if (adType == "ssa") {
      val adImpAndClickMap: Map[String, (Double, Double)]
       = _getIdImpressionAndClickMap("ad_id", statDataDF)

      // Positional Bias Map
      val positionBiasMap: Map[String, Double] = _getIdCTRMap("total_rank", statDataDF)

      // ad x id CTR map
      val adxChannelIdCTRMap: Map[String, Double] = _getPairIdCTRMap("ad_id", "channel_id", statDataDF)
      val adxMediaUIIdCTRMap: Map[String, Double] = _getPairIdCTRMap("ad_id", "media_ui_type_code", statDataDF)

      val bcPositionBiasMap = sc.broadcast(positionBiasMap)
      val bcAdImpAndClickMap = sc.broadcast(adImpAndClickMap)
      
      // ad stat map (ad -> (coec, imp, click))
      val statMap: Map[String, (Double, Double, Double)] =
          _getAdStatisicalMap(statDataDF, "ad_id", bcPositionBiasMap, bcAdImpAndClickMap)

      // destory a broadcast, statCount
      bcAdImpAndClickMap.destroy()

      (statMap, positionBiasMap, adxChannelIdCTRMap, adxMediaUIIdCTRMap) 

    } else if (adType == "sa") {
      val pairIdImpAndClickMap : Map[String, (Double, Double)] = 
        _getPairIdImpressionAndClipMap("ad_id", "sch_keyword", statDataDF)

      // Positional Bias Map
      val positionBiasMap: Map[String, Double] = _getIdCTRMap("total_rank", statDataDF)
      
      val bcPositionBiasMap = sc.broadcast(positionBiasMap)
      val bcPairIdImpAndClickMap = sc.broadcast(pairIdImpAndClickMap)

      // ad stat map (adxkeyword -> (coec, imp, click))
      val statMap: Map[String, (Double, Double, Double)] =
          _getAdxIdStatisicalMap(statDataDF, "ad_id", "sch_keyword", bcPositionBiasMap, bcPairIdImpAndClickMap)

      (statMap, positionBiasMap, Map[String, Double](), Map[String, Double]())
    } else {
       (Map[String, (Double, Double, Double)](), Map[String, Double](),
         Map[String, Double](), Map[String, Double]())
    }
  }

}
