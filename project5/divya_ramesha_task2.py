from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import binascii
import sys
import json


def getTrailingZeros(n):
    s = str(bin(n))[2:]
    return len(s) - len(s.rstrip('0'))


if __name__ == "__main__":

    hashes = [[68, 80, 211], [55, 62, 269], [19, 30, 277], [10, 52, 157], [52, 49, 181], [27, 96, 197],
              [22, 72, 199], [71, 16, 173], [41, 13, 233], [27, 65, 179], [48, 45, 149], [28, 81, 107]]


    def flajoletMartin(time, rdd):
        finalCountsRecorded = [0] * hashesLength
        citiesList = rdd.collect()
        uniqueCitiesSet = set(citiesList)
        flojoletUniqueCount = 0

        if len(citiesList) > 0:

            for city in citiesList:
                cityInt = int(binascii.hexlify(city.encode('utf8')), 16)
                for index, hash in enumerate(hashes):
                    hashVal = (hash[0] * cityInt + hash[1]) % hash[2]
                    trailingZeros = getTrailingZeros(hashVal)
                    finalCountsRecorded[index] = max(finalCountsRecorded[index], trailingZeros)

            finalCountsRecordedWith2Power = []
            for num in finalCountsRecorded:
                finalCountsRecordedWith2Power.append(2**num)

            group1 = sum(finalCountsRecordedWith2Power[:hashesLengthHalf]) / hashesLengthHalf
            group2 = sum(finalCountsRecordedWith2Power[hashesLengthHalf:]) / hashesLengthHalf
            flojoletUniqueCount = (group1 + group2) / 2

        with open(outputFile, "a") as outFile:
            outFile.write("\n" + str(time) + "," + str(len(uniqueCitiesSet)) + "," + str(flojoletUniqueCount))


    portNumber = int(sys.argv[1])
    outputFile = sys.argv[2]

    with open(outputFile, "w") as outFile:
        outFile.write("Time,Ground Truth,Estimation")

    spark = SparkSession.builder.appName('inf-553-5b').getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 5)
    rdd = ssc.socketTextStream("localhost", portNumber)

    hashesLength = len(hashes)
    hashesLengthHalf = hashesLength // 2

    rdd = rdd.window(30, 10)
    citiesRdd = rdd.map(lambda jsn: json.loads(jsn)["city"])
    citiesRdd.foreachRDD(flajoletMartin)

    ssc.start()
    ssc.awaitTermination()