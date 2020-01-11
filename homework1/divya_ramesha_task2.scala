import java.io.{BufferedWriter, File, FileWriter}

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.{compact, parse}

import scala.collection.mutable
import scala.reflect.classTag

object divya_ramesha_task2 {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val ss = SparkSession.builder().appName("inf-553-1b").config("spark.master", "local[*]").getOrCreate()
    val sc = ss.sparkContext

    val jsonRDD = sc.textFile(args(0))

    val s1 = System.nanoTime

    val top10_most_reviews1 = jsonRDD.map{row =>
      val json = parse(row)
      (compact(json \ "user_id"), compact(json \ "review_count").toInt)}.sortBy(x => (x._2, x._1))(Ordering.Tuple2(Ordering.Int.reverse, Ordering.String), classTag[(Int,String)]).take(10)

    val e1 = System.nanoTime

    val m1 = jsonRDD.mapPartitions(iter => Array(iter.size).iterator, true).collect()

    val newjsonRDD = jsonRDD.repartition(args(2).toInt)

    val s2 = System.nanoTime

    val top10_most_reviews2 = newjsonRDD.map{row =>
      val json = parse(row)
      (compact(json \ "user_id"), compact(json \ "review_count").toInt)}.sortBy(x => (x._2, x._1))(Ordering.Tuple2(Ordering.Int.reverse, Ordering.String), classTag[(Int,String)]).take(10)

    val e2 = System.nanoTime

    val m2 = newjsonRDD.mapPartitions(iter => Array(iter.size).iterator, true).collect()

    val exe1 = (e1 - s1) / 1e9d
    val exe2 = (e2 - s2) / 1e9d

    val task2 = mutable.LinkedHashMap(
      "default" -> mutable.LinkedHashMap(
          "n_partition" -> m1.length,
          "n_items" -> m1,
          "exe_time" -> exe1
      ),
      "customized"-> mutable.LinkedHashMap(
          "n_partition" -> m2.length,
          "n_items" -> m2,
          "exe_time" -> exe2
      ),
      "explanation"-> "If all the elements are distributed properly to Mapper with less number of partitions then it takes optimal time. Very less or very large partition size affects the complexity badly. Very less means more burden on tasks. Very large means more overhead on communication between tasks."
    )

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val file = new File(args(1))
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(mapper.writeValueAsString(task2))
    bw.close()

  }

}
