from pyspark.sql import SparkSession
import sys
from operator import add
import time
import json

spark = SparkSession.builder.appName('inf-553-1a').getOrCreate()
sc = spark.sparkContext

start_time = time.time()

df = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))

task1 = {}

task1["total_users"] = df.count()

task1["avg_reviews"] = df.map(lambda user: user['review_count']).sum() / task1["total_users"]

task1["distinct_usernames"] = df.map(lambda user: user['name']).distinct().count()

task1["num_users"] = df.filter(lambda user: user['yelping_since'].split("-")[0] == "2011").count()

task1["top10_popular_names"] = df.map(lambda usr: (usr['name'], 1)).reduceByKey(add).sortByKey().sortBy(lambda x: x[1], ascending=False).take(10)

task1["top10_most_reviews"] = df.map(lambda user: (user['user_id'], user['review_count'])).sortByKey().sortBy(lambda x: x[1], ascending=False).take(10)

with open(sys.argv[2], 'w') as outfile:
  json.dump(task1, outfile)