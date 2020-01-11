from pyspark.sql import SparkSession
import sys
from operator import add
import time
import json

spark = SparkSession.builder.appName('inf-553-1c').getOrCreate()
sc = spark.sparkContext

start_time = time.time()

r = sc.textFile(sys.argv[1]).map(lambda x: json.loads(x))
b = sc.textFile(sys.argv[2]).map(lambda x: json.loads(x))

r1 = r.map(lambda i: (i['business_id'], i['stars']))
b1 = b.map(lambda i: (i['business_id'], i['state']))

intermediate_result = b1.join(r1).map(lambda i: (i[1][0], i[1][1])).groupByKey().mapValues(lambda v: sum(v)/len(v)).sortByKey().sortBy(lambda x: x[1], ascending=False)

s3 = time.time()

res = intermediate_result.collect()

for index in range(5):
  print(res[index])

e3 = time.time()

with open(sys.argv[3], 'w') as outfile:
  outfile.write("state,stars")
  for i in res:
    outfile.write("\n" + i[0] + "," + str(i[1]))

s4 = time.time()

res = intermediate_result.take(5)
print(res)

e4 = time.time()

task3 = {}
task3["m1"] = e3 - s3
task3["m2"] = e4 - s4
task3["explanation"] = "take() method works faster because it'll return the result as soon as it scans the first 5 elements whereas collect() scans through all the elements"

with open(sys.argv[4], 'w') as outfile:
  json.dump(task3, outfile)