from divya_ramesha_functions import pairs_by_lsh
from pyspark.sql import SparkSession
import time
import sys


if __name__ == "__main__":

	start = time.time()

	inputFile = sys.argv[1]
	similarityMethod = sys.argv[2]
	outputFile = sys.argv[3]

	if similarityMethod == "jaccard":

		spark = SparkSession.builder.appName('inf-553-3b').getOrCreate()
		sc = spark.sparkContext

		trainRdd = sc.textFile(inputFile)
		trainHeader = trainRdd.first()
		trainDataset = trainRdd.filter(lambda row: row != trainHeader).map(lambda x: x.split(","))

		similarPairs = pairs_by_lsh(trainDataset).sortBy(lambda x: x[1]).sortBy(lambda x: x[0])

		with open(outputFile, "w") as outFile:
			outFile.write("business_id_1, business_id_2, similarity")
			for pair in similarPairs.collect():
				outFile.write("\n" + pair[0] + "," + pair[1] + "," + str(pair[2]))

		print("Duration: ", (time.time() - start))

	else:
		print("Sorry! I didn't had time to implement!")
