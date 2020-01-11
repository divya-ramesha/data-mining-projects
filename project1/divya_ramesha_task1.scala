import java.io.{BufferedWriter, File, FileWriter}

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.{compact, parse}

import scala.collection.mutable
import scala.reflect.classTag

object divya_ramesha_task1 {

    def main(args: Array[String]): Unit = {

      Logger.getLogger("org").setLevel(Level.ERROR)

      val ss = SparkSession.builder().appName("inf-553-1a").config("spark.master", "local[*]").getOrCreate()
      val sc = ss.sparkContext

      val jsonRDD = sc.textFile(args(0))

      val total_users = jsonRDD.count()
      val total_reviews = jsonRDD.map{row => compact(parse(row) \ "review_count").toInt}.sum()
      val distinct_usernames = jsonRDD.map{row => compact(parse(row) \ "name")}.distinct().count()
      val num_users = jsonRDD.filter(row => compact(parse(row) \ "yelping_since").replace("\"", "").split("-")(0) == "2011").count()
      val top10_popular_names = jsonRDD.map{row => (compact(parse(row) \ "name"), 1)}.reduceByKey(_+_).sortBy(x => (x._2, x._1))(Ordering.Tuple2(Ordering.Int.reverse,Ordering.String), classTag[(Int,String)]).take(10)
      val top10_most_reviews = jsonRDD.map{row =>
        val json = parse(row)
        (compact(json \ "user_id"), compact(json \ "review_count").toInt)}.sortBy(x => (x._2, x._1))(Ordering.Tuple2(Ordering.Int.reverse, Ordering.String), classTag[(Int,String)]).take(10)

      val task1 = mutable.LinkedHashMap(
        "total_users" -> total_users,
        "avg_reviews" -> total_reviews / total_users,
        "distinct_usernames" -> distinct_usernames,
        "num_users" -> num_users,
        "top10_popular_names" -> top10_popular_names,
        "top10_most_reviews" -> top10_most_reviews)

      val mapper = new ObjectMapper()
      mapper.registerModule(DefaultScalaModule)

      val file = new File(args(1))
      val bw = new BufferedWriter(new FileWriter(file))
      bw.write(mapper.writeValueAsString(task1).replaceAll("\\\\\"", ""))
      bw.close()
    }

}
