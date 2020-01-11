from pyspark.sql import SparkSession
from operator import add
from itertools import combinations
import csv
import sys
import time


def candidateCount(partitionBasket, candidateItems):
    itemCount = {}

    for candidate in candidateItems:
        if type(candidate) is int:
            candidate = [candidate]
        item = tuple(sorted(candidate))
        candidate = set(candidate)
        for basket in partitionBasket:
            if candidate.issubset(basket):
                itemCount[item] = itemCount.get(item, 0) + 1

    return itemCount


def getFrequentItems(partitionBasket, candidateItems, basketSupport):
    itemCount = candidateCount(partitionBasket, candidateItems)
    filteredFrequentItems = [item for item in itemCount if itemCount[item] >= basketSupport]
    return filteredFrequentItems


def getCandidateItems(frequentItems, k):

    if k == 2:
        return list(combinations(frequentItems, 2))

    candidateItems = []

    for item in combinations(frequentItems, 2):
        new_item = set(item[0] + item[1])
        new_item_tuple = tuple(sorted(new_item))
        if len(new_item) == k and new_item_tuple not in candidateItems:
            flag = True
            for sub_item in combinations(new_item, k - 1):
                if tuple(sorted(sub_item)) not in frequentItems:
                    flag = False
                    break
            if flag:
                candidateItems.append(new_item_tuple)

    return candidateItems


def aprioriAlgorithm(partitionBasket, support, totalCount):

    partitionBasket = list(partitionBasket)
    basketSupport = support * (float(len(partitionBasket)) / float(totalCount))
    finalResult = list()

    candidateItems = []
    for basket in partitionBasket:
        candidateItems += list(basket)

    frequentItems = []
    for item in set(candidateItems):
        if candidateItems.count(item) >= basketSupport:
            frequentItems.append(item)

    frequentItems = sorted(frequentItems)

    finalResult.extend(frequentItems)

    k = 2
    while len(frequentItems) > 0:

        candidateItems = getCandidateItems(frequentItems, k)

        if len(candidateItems) > 0:
            frequentItems = getFrequentItems(partitionBasket, candidateItems, basketSupport)
            frequentItems = sorted(frequentItems)
            finalResult.extend(frequentItems)
        else:
            frequentItems = []

        k = k + 1

    return finalResult


def writeItemsToFile(outputItems, outFilePtr):

    length = 1
    output = {}

    nextList = list(filter(lambda l: len(l) == length, outputItems))

    while nextList:
        output[length] = nextList
        length += 1
        nextList = list(filter(lambda l: len(l) == length, outputItems))

    for key in sorted(output.keys()):

        outputList = output[key]
        outputItems = []
        for item in outputList:
            outputItems.append(tuple(sorted(str(i) for i in item)))

        outputItems = sorted(outputItems)

        if key == 1:
            outFilePtr.write(str(outputItems[0]).replace(",", ""))
            for i in outputItems[1:]:
                outFilePtr.write(",")
                outFilePtr.write(str(i).replace(",", ""))
        else:
            outFilePtr.write("\n\n")
            outFilePtr.write(str(outputItems[0]))
            for i in outputItems[1:]:
                outFilePtr.write(",")
                outFilePtr.write(str(i))


def removeLeadingZeros(num):
    while num[0] == "0":
        num = num[1:]
    return int(num)


def update_year(date_string):
    date_list = date_string.split("/")
    if len(date_list[2]) == 4:
        date_string = "/".join(date_list[:2]) + "/" + date_list[2][-2:]
    return date_string


if __name__ == "__main__":

    start = time.time()

    filterThreshold = int(sys.argv[1])
    support = int(sys.argv[2])
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]

    outFile = open(outputFile, "w")

    spark = SparkSession.builder.appName('inf-553-2b').getOrCreate()
    sc = spark.sparkContext

    dataset = sc.textFile(inputFile).mapPartitions(lambda x : csv.reader(x))

    # Data PreProcessing
    header = dataset.first()
    transactionIndex = header.index("TRANSACTION_DT")
    customerIndex = header.index("CUSTOMER_ID")
    productIndex = header.index("PRODUCT_ID")
    dataset = dataset.filter(lambda x: x != header)

    dataset = dataset.map(lambda x: (update_year(x[transactionIndex]) + "-" + x[customerIndex], removeLeadingZeros(x[productIndex])))
    with open('divya_ramesha_preprocessed.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['DATE-CUSTOMER_ID', 'PRODUCT_ID']] + dataset.collect())

    # Filter Baskets
    marketBaskets = dataset.groupByKey().mapValues(set).filter(lambda x: len(x[1]) > filterThreshold)
    marketBaskets = marketBaskets.map(lambda x: x[1])
    basketsCount = marketBaskets.count()

    # First Map
    firstMap = marketBaskets.mapPartitions(lambda partitionBasket: aprioriAlgorithm(partitionBasket, support, basketsCount)).map(lambda x: (x, 1))

    # First Reduce
    firstReduce = firstMap.reduceByKey(add).keys()

    outFile.write("Candidates:\n")
    singleItems = firstReduce.filter(lambda x: type(x) == int).collect()
    it = iter(singleItems)
    singleItems = list(zip(it))
    otherItems = firstReduce.filter(lambda x: type(x) != int).collect()
    writeItemsToFile(singleItems + otherItems, outFile)

    candidateKeys = firstReduce.collect()
    # Second Map
    secondMap = marketBaskets.mapPartitions(lambda partitionBasket: candidateCount(list(partitionBasket), candidateKeys).items())
    # Second Reduce
    secondReduce = secondMap.reduceByKey(add).filter(lambda x: x[1] >= support)

    outFile.write("\n\nFrequent Itemsets:\n")
    writeItemsToFile(secondReduce.keys().collect(), outFile)

    outFile.close()
    end = time.time()
    print("Duration: ", end - start)