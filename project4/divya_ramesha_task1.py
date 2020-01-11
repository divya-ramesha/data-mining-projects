from pyspark.sql import SQLContext, SparkSession
from operator import add
from graphframes import *
import time
import sys


if __name__ == "__main__":

    start = time.time()

    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-4a').getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    edges = sc.textFile(inputFile).map(lambda x: x.split(" ")).flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])])
    edgesDf = spark.createDataFrame(edges).toDF("src", "dst")

    verticesDf = edgesDf.select(edgesDf['src'].alias('id')).distinct()

    networkGraph = GraphFrame(verticesDf, edgesDf)
    result = networkGraph.labelPropagation(maxIter=5)
    communities = result.rdd.map(lambda x: (x[1], [str(x[0])])).reduceByKey(add).map(lambda x: sorted(x[1])).collect()

    sortedCommunities = sorted(communities, key=lambda l: (len(l), l))
    totalcommunities = len(sortedCommunities)
    
    for index, community in enumerate(sortedCommunities):
        outFile.write("'" + community[0] + "'")
        if len(community) > 1:
            for node in community[1:]:
                outFile.write(", ")
                outFile.write("'" + node + "'")
        if index != totalcommunities - 1:
            outFile.write("\n")

    outFile.close()
    print("Duration: ", time.time() - start)