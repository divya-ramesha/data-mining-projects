import os
import sys
import time
from pyspark.sql import SparkSession
from surprise import Dataset, Reader, BaselineOnly


def cutPrediction(x):
    if x[1] > 5:
        rate = 5
    elif x[1] < 1:
        rate = 1
    else:
        rate = x[1]
    return (x[0][0], x[0][1]), rate


if __name__ == "__main__":

    start = time.time()

    trainFilePath = sys.argv[1] + "yelp_train.csv"
    testFilePath = sys.argv[2]
    outputFile = sys.argv[3]

    outFile = open(outputFile, "w")

    trainFilePath = os.path.expanduser(trainFilePath)

    spark = SparkSession.builder.appName('inf-553-project').getOrCreate()
    sc = spark.sparkContext

    testRdd = sc.textFile(testFilePath)
    testHeader = testRdd.first()
    testDataset = testRdd.filter(lambda row: row != testHeader).map(lambda x: x.split(","))
    testDatasetPredictions = testDataset.map(lambda x: ((x[0], x[1]), float(x[2])))

    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)

    bsl_options = {
        'method': 'als',
        'n_epochs': 20,
        'reg_u': 6,
        'reg_i': 4
        }

    trainSet = Dataset.load_from_file(trainFilePath, reader=reader).build_full_trainset()

    algo1 = BaselineOnly(bsl_options=bsl_options, verbose=False)
    algo1.fit(trainSet)

    calculatedPredictions = testDatasetPredictions.map(lambda x: ((x[0][0], x[0][1]), algo1.predict(x[0][0], x[0][1], r_ui=x[1], verbose=False).est)).map(cutPrediction)

    outFile.write("user_id, business_id, prediction")
    for data in calculatedPredictions.collect():
        outFile.write("\n" + data[0][0] + "," + data[0][1] + "," + str(data[1]))
    outFile.close()

    ratingsDifference = testDatasetPredictions.join(calculatedPredictions).map(lambda x: abs(x[1][0] - x[1][1]))

    diff01 = ratingsDifference.filter(lambda x: 0 <= x < 1).count()
    diff12 = ratingsDifference.filter(lambda x: 1 <= x < 2).count()
    diff23 = ratingsDifference.filter(lambda x: 2 <= x < 3).count()
    diff34 = ratingsDifference.filter(lambda x: 3 <= x < 4).count()
    diff4 = ratingsDifference.filter(lambda x: 4 <= x).count()

    MSE = ratingsDifference.map(lambda x: x ** 2).mean()
    RMSE = pow(MSE, 0.5)

    print("\n>=0 and <1:", diff01)
    print(">=1 and <2:", diff12)
    print(">=2 and <3:", diff23)
    print(">=3 and <4:", diff34)
    print(">=4:", diff4)
    print("\nRMSE: ", RMSE)
    print("Execution Time: ", time.time() - start)
