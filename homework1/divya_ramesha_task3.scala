import java.io.{BufferedWriter, File, FileWriter}
import scala.reflect.classTag
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.{compact, parse}

import scala.collection.mutable

object divya_ramesha_task3 {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val ss = SparkSession.builder().appName("inf-553-1c").config("spark.master", "local[*]").getOrCreate()
    val sc = ss.sparkContext

    val reviewRDD = sc.textFile(args(0)).map{row =>
                val json = parse(row)
                (compact(json \ "business_id"), compact(json \ "stars").toDouble)}

    val businessRDD = sc.textFile(args(1)).map{row =>
                val json = parse(row)
                (compact(json \ "business_id"), compact(json \ "state"))}


    val intermediateResult = businessRDD.join(reviewRDD).map{i => (i._2._1, i._2._2)}.groupByKey().mapValues(v => v.sum / v.size).sortBy(x => (x._2, x._1))(Ordering.Tuple2(Ordering.Double.reverse,Ordering.String), classTag[(Double,String)])

    val s3 = System.nanoTime

    val res1 = intermediateResult.collect()

    for( i <- 0 to 4){
      println(res1(i));
    }

    val e3 = System.nanoTime

    val file = new File(args(2))
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("state,stars")

    res1.foreach{ x =>
      bw.write("\n" + x._1.replace("\"", "") + "," + x._2)
    }

    bw.close()

    val s4 = System.nanoTime

    val res2 = intermediateResult.take(5)
    res2.foreach{ println }

    val e4 = System.nanoTime

    val exe1 = (e3 - s3) / 1e9d
    val exe2 = (e4 - s4) / 1e9d

    val task3 = mutable.LinkedHashMap(
      "m1" -> exe1,
      "m2" -> exe2,
      "explanation" -> "take() method works faster because it'll return the result as soon as it scans the first 5 elements whereas collect() scans through all the elements"
    )

    val mapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)

    val file2 = new File(args(3))
    val bw2 = new BufferedWriter(new FileWriter(file2))
    bw2.write(mapper.writeValueAsString(task3))
    bw2.close()

  }

}
