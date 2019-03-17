package com.naver.matching

import java.text.SimpleDateFormat
import java.util.Calendar

case class CommandLineArgs(
  adType: String = "ssa",
  dataType: String = "train",
  saveDir: String = "",
  stringToIntDir: String = "",
  negativeSamplingRate: Double = 5.0,
  samplingRate: Double = 0.5,
  fromDate: String = "",
  toDate: String = "",
  inferDate: String = "",
  genStringIndexer: Boolean = false,
  period: Int = 14,
  bucketId: String = "") {
  val dateFormat = new SimpleDateFormat("yyyyMMdd")

  def getToDate = {
    if (toDate.isEmpty) {
      val calendar = Calendar.getInstance()
      calendar.add(Calendar.DATE, -1) 
      dateFormat.format(calendar.getTime)
    } else
       toDate
  }

  def getFromDate = {
    if (fromDate.isEmpty){
      val calendar = Calendar.getInstance()
      calendar.add(Calendar.DATE, -period)
      dateFormat.format(calendar.getTime)
    } else
      fromDate
  }

  def getInferDate = {
    if (inferDate.isEmpty) {
      val calendar = Calendar.getInstance()
      val hourOfDay = calendar.get(Calendar.HOUR_OF_DAY)
      if (hourOfDay < 1) {
        calendar.add(Calendar.DATE, -1)
        dateFormat.format(calendar.getTime)
      } else
        dateFormat.format(calendar.getTime)
    } else
      inferDate
  }

  def getStatToDate = {
    if (dataType == "infer") {
      getInferDate
    } else 
      getToDate
  }

  def getStatFromDate = {
    val toDate = dateFormat.parse(getStatToDate)
    val calendar = Calendar.getInstance()
    calendar.setTime(toDate)
    calendar.add(Calendar.DATE, -period)
    dateFormat.format(calendar.getTime)
  }
}
