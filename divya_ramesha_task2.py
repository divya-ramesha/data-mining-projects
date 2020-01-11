from pyspark.sql import SparkSession
import sys
from operator import add
import time
import json

spark = SparkSession.builder.appName('inf-553-1b').getOrCreate()
sc = spark.sparkContext

start_time = time.time()

df = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))

s1 = time.time()

task21 = df.map(lambda user: (user['review_count'], user['user_id'])).sortBy(lambda x: x[1]).sortByKey(ascending=False).take(10)

e1 = time.time()

m1 = df.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()

df = df.repartition(int(sys.argv[3]))

s2 = time.time()

top10_most_reviews = df.map(lambda user: [user['review_count'], user['user_id']]).sortBy(lambda x: x[1]).sortByKey(ascending=False).take(10)

e2 = time.time()

m2 = df.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()

task2 = {}

task2["default"] = {
  "n_partition": len(m1),
  "n_items": m1,
  "exe_time": e1 - s1
}

task2["customized"] = {
  "n_partition": len(m2),
  "n_items": m2,
  "exe_time": e2 - s2
}

task2["explanation"] = "If all the elements are distributed properly to Mapper with less number of partitions then it takes optimal time. Very less or very large partition size affects the complexity badly. Very less means more burden on tasks. Very large means more overhead on communication between tasks."

with open(sys.argv[2], 'w') as outfile:
  json.dump(task2, outfile)