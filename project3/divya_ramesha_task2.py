from divya_ramesha_functions import model_based_cf, used_based_cf, item_based_cf
import sys


if __name__ == "__main__":

    trainFilePath = sys.argv[1]
    testFilePath = sys.argv[2]
    caseId = int(sys.argv[3])
    outputFile = sys.argv[4]

    if caseId == 1:
        model_based_cf(trainFilePath, testFilePath, outputFile)

    elif caseId == 2:
        used_based_cf(trainFilePath, testFilePath, outputFile)

    elif caseId == 3:
        item_based_cf(trainFilePath, testFilePath, outputFile)
