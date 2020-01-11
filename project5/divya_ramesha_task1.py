from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import binascii
import json
import sys


if __name__ == "__main__":

    hashes = [[3, 19], [29, 53], [13, 47], [23, 61], [37, 71], [89, 139], [107, 193], [163, 199]]


    def bloomFilter(time, rdd):
        global falsePositiveCount, trueNegativeCount, globalStateSet, globalBitArray, hashes
        statesList = rdd.collect()
        if len(statesList) > 0:
            for state in statesList:
                stateInt = int(binascii.hexlify(state.encode('utf8')), 16)

                hashValues = []
                for i in range(len(hashes)):
                    hashValues.append((hashes[i][0] * stateInt + hashes[i][1]) % 200)

                bloomDuplicate = True
                for hashVal in hashValues:
                    if globalBitArray[hashVal] == 0:
                        bloomDuplicate = False
                        break

                if bloomDuplicate:
                    if state not in globalStateSet:
                        falsePositiveCount += 1
                else:
                    if state not in globalStateSet:
                        trueNegativeCount += 1

                for hashVal in hashValues:
                    globalBitArray[hashVal] = True
                globalStateSet.add(state)
        falsePositiveRate = 0.0
        if falsePositiveCount > 0:
            falsePositiveRate = falsePositiveCount / (falsePositiveCount + trueNegativeCount)

        with open(outputFile, "a") as outFile:
            outFile.write("\n" + str(time) + "," + str(falsePositiveRate))


    portNumber = int(sys.argv[1])
    outputFile = sys.argv[2]

    with open(outputFile, "w") as outFile:
        outFile.write("Time,FPR")

    spark = SparkSession.builder.appName('inf-553-5a').getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 10)
    rdd = ssc.socketTextStream("localhost", portNumber)
    globalBitArray = [False] * 200
    globalStateSet = set()

    falsePositiveCount, trueNegativeCount = 0, 0

    statesRdd = rdd.map(lambda jsn: json.loads(jsn)["state"])
    statesRdd.foreachRDD(bloomFilter)

    ssc.start()
    ssc.awaitTermination()