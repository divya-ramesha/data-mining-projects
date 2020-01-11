from pyspark.sql import SparkSession
from operator import add
from itertools import combinations
import time
import copy
import sys


def calculateEdgeWeights(adjacencyMap, vertexWeightsMap, reversedlevelOrderTraversal):
    edgeWeightsMap = {}
    for index, currentLevel in enumerate(reversedlevelOrderTraversal[:-1]):
        for vertex in currentLevel:
            splitWeight = 1
            if index != 0:
                connectedChildNodes = adjacencyMap[vertex].intersection(reversedlevelOrderTraversal[index - 1])
                splitWeight = 1 + sum([edgeWeightsMap[(min(vertex, childNode), max(vertex, childNode))] for childNode in connectedChildNodes])
            connectedParentNodes = adjacencyMap[vertex].intersection(reversedlevelOrderTraversal[index + 1])
            vertexWeight = sum([vertexWeightsMap[parentNode] for parentNode in connectedParentNodes])
            for parentNode in connectedParentNodes:
                edgeWeight = (float(vertexWeightsMap[parentNode])/float(vertexWeight)) * float(splitWeight)
                edgeWeightsMap[(min(vertex, parentNode), max(vertex, parentNode))] = edgeWeight

    return edgeWeightsMap


def calculateVertexWeights(adjacencyMap, levelOrderTraversal):
    vertexWeights = {}
    levels = len(levelOrderTraversal)
    if levels >= 2:
        for user_id in levelOrderTraversal[1]:
            vertexWeights[user_id] = 1

    for currentLevel in range(2, levels):
        currentLevelNodes = levelOrderTraversal[currentLevel]
        parentLevelNodes = levelOrderTraversal[currentLevel - 1]
        for vertex in currentLevelNodes:
            connectedParentNodes = adjacencyMap[vertex].intersection(parentLevelNodes)
            vertexWeights[vertex] = sum([vertexWeights[parentVertex] for parentVertex in connectedParentNodes])

    return vertexWeights


def bfs(vertex, verticesCount, adjacencyMap):
    visitedVertices = [False for _ in range(verticesCount + 1)]
    currentLevel, nextLevel = [vertex], []
    visitedVertices[vertex] = True
    levelOrderTraversal = []
    verticesInCurrentLevel = []
    while currentLevel:
        currentVertex = currentLevel.pop(0)
        for adjacentVertex in adjacencyMap[currentVertex]:
            if not visitedVertices[adjacentVertex]:
                visitedVertices[adjacentVertex] = True
                nextLevel.append(adjacentVertex)
        verticesInCurrentLevel.append(currentVertex)

        if not currentLevel:
            currentLevel = nextLevel
            nextLevel = []
            levelOrderTraversal.append(verticesInCurrentLevel)
            verticesInCurrentLevel = []
    return levelOrderTraversal


def calculateBetweenness(vertices, verticesCount, adjacencyMap):
    allEdgeWeights = dict()
    for vertex in vertices:
        levelOrderTraversal = bfs(vertex, verticesCount, adjacencyMap)
        vertexWeights = calculateVertexWeights(adjacencyMap, levelOrderTraversal)
        vertexWeights[vertex] = 1
        currentEdgeWeights = calculateEdgeWeights(adjacencyMap, vertexWeights, list(reversed(levelOrderTraversal)))
        for edge in currentEdgeWeights:
            allEdgeWeights[edge] = allEdgeWeights.get(edge, 0) + currentEdgeWeights[edge]
    yield allEdgeWeights.items()


def getAllConnectedGroups(verticesList, adjacencyMap):
    already_seen = set()
    result = []
    for vertex in verticesList:
        if vertex not in already_seen:
            connected_group = []
            nodes = {vertex}
            while nodes:
                node = nodes.pop()
                already_seen.add(node)
                if node in adjacencyMap:
                    unexploredNodes = adjacencyMap[node] - already_seen
                    if len(unexploredNodes) > 0:
                        nodes = nodes | unexploredNodes
                connected_group.append(node)
            result.append(connected_group)
    return result


if __name__ == "__main__":

    start = time.time()

    inputFile = sys.argv[1]
    betweennessFile = sys.argv[2]
    communityFile = sys.argv[3]

    betweennessFilePtr = open(betweennessFile, "w")
    communityFilePtr = open(communityFile, "w")

    spark = SparkSession.builder.appName('inf-553-4b').getOrCreate()
    sc = spark.sparkContext

    edges = sc.textFile(inputFile).map(lambda x: x.split(" "))
    adjacencyList = edges.flatMap(lambda x: [(int(x[0]), {int(x[1])}), (int(x[1]), {int(x[0])})]).reduceByKey(lambda x, y: x | y)

    adjacencyMap = {}
    for v in adjacencyList.collect():
        adjacencyMap[v[0]] = v[1]
    adjacencyMapDeepCopy = copy.deepcopy(adjacencyMap)

    graphVertices = adjacencyList.keys().collect()
    verticesCount = len(adjacencyMap)

    betweennessEdges = sc.parallelize(graphVertices).mapPartitions(lambda vertices: calculateBetweenness(vertices, verticesCount, adjacencyMap)).flatMap(list).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)).sortBy(lambda x: -x[1])
    betweennessTuples = betweennessEdges.map(lambda x: (sorted([str(x[0][0]), str(x[0][1])]), x[1])).sortByKey().sortBy(lambda x: -x[1]).collect()
    betweennessLength = len(betweennessTuples)

    if betweennessLength > 0:
        for i, e in enumerate(betweennessTuples):
            betweennessFilePtr.write("('" + str(e[0][0]) + "', '" + str(e[0][1]) + "'), " + str(e[1]))
            if i != betweennessLength - 1:
                betweennessFilePtr.write("\n")
    betweennessFilePtr.close()

    vertexDegree = {}
    for node in graphVertices:
        vertexDegree[node] = len(adjacencyMap[node])

    modularity, edgeCount = 0, edges.count()
    modularityMap = {}
    for verticesPair in combinations(graphVertices, 2):
        vertexI = min(verticesPair[0],  verticesPair[1])
        vertexJ = max(verticesPair[0],  verticesPair[1])
        if vertexJ in adjacencyMap[vertexI]:
             A = 1
        else:
            A = 0
        q = (A - ((0.5 * vertexDegree[vertexI] * vertexDegree[vertexJ]) / edgeCount))
        modularityMap[(vertexI, vertexJ)] = q
        modularity += q

    maxModularity = (1 / (2 * edgeCount)) * modularity
    finalCommunities = getAllConnectedGroups(graphVertices, adjacencyMap)

    for edgeIndex in range(1, edges.count() + 1):

        betweenTuple = betweennessEdges.take(1)[0]
        i = int(betweenTuple[0][0])
        j = int(betweenTuple[0][1])

        adjacencyMap[i].remove(j)
        adjacencyMap[j].remove(i)

        connectedCommunities = getAllConnectedGroups(graphVertices, adjacencyMap)

        currentModularity = 0
        currentCommunities = []

        for community in connectedCommunities:
            communityModularity = 0
            for verticesPair in combinations(community, 2):
                vertexI = min(verticesPair[0], verticesPair[1])
                vertexJ = max(verticesPair[0], verticesPair[1])
                communityModularity += modularityMap[(vertexI, vertexJ)]
            currentCommunities.append(sorted(community))
            currentModularity += communityModularity
        currentModularity = (1 / (2 * edgeCount)) * currentModularity

        if currentModularity > maxModularity:
            maxModularity = currentModularity
            finalCommunities = currentCommunities

        betweennessEdges = sc.parallelize(graphVertices).mapPartitions(lambda vertices: calculateBetweenness(vertices, verticesCount, adjacencyMap)).flatMap(list).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)).sortBy(lambda x: -x[1])

    finalCommunities = [sorted([str(i) for i in community]) for community in finalCommunities]
    finalCommunities = sorted(finalCommunities, key=lambda l: (len(l), l))
    totalcommunities = len(finalCommunities)

    for index, community in enumerate(finalCommunities):
        communityFilePtr.write("'" + community[0] + "'")
        if len(community) > 1:
            for node in community[1:]:
                communityFilePtr.write(", ")
                communityFilePtr.write("'" + node + "'")
        if index != totalcommunities - 1:
            communityFilePtr.write("\n")
    communityFilePtr.close()

    print("Duration: ", time.time() - start)