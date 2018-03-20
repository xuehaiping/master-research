import math


def dotProduct(v1, v2):
    summation = 0
    for i in range(len(v1)):
        summation += v1[i]*v2[i]
    return summation


def magnitude(v1):
    summation = 0
    for ele in v1:
        summation += ele*ele
    return math.sqrt(summation)


def cosineSim(v1, v2):
    return dotProduct(v1, v2)/(magnitude(v1)* magnitude(v2))