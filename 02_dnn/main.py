import cv2
import numpy as np
import math


def conv(img, kernels, bias):
    result = np.full(
        (kernels.shape[0], img.shape[0], img.shape[1]), bias, np.float32)
    for m in range(kernels.shape[0]):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for i in range(kernels.shape[1]):
                    for j in range(kernels.shape[2]):
                        for c in range(kernels.shape[3]):
                            if (x + i < img.shape[0]) and (y + j < img.shape[1]):
                                result[m, x, y] += img[x + i, y + j, c] * \
                                    kernels[m, i, j, c]
    return result


def normalize(arr, gamma=1.0, bias=0.0):
    M = np.zeros((arr.shape[0]), dtype=np.float32)
    for m in range(arr.shape[0]):
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                M[m] += arr[m, i, j]
    M /= arr.shape[1] * arr.shape[2]
    D = np.zeros((arr.shape[0]), dtype=np.float32)
    for m in range(arr.shape[0]):
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                D[m] += (arr[m, i, j] - M[m]) ** 2
    D /= arr.shape[1] * arr.shape[2]
    result = np.zeros(arr.shape, dtype=np.float32)
    for m in range(result.shape[0]):
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                result[m, i, j] = gamma * \
                    (arr[m, i, j] - M[m]) / math.sqrt(D[m]) + bias
    return result


def relu(arr):
    result = np.zeros(arr.shape, dtype=np.float32)
    for m in range(result.shape[0]):
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                result[m, i, j] = max(0, arr[m, i, j])
    return result


def maxpooling(arr):
    result = np.zeros(
        (arr.shape[0], int(arr.shape[1] / 2), int(arr.shape[2] / 2)), dtype=np.float32)
    for m in range(result.shape[0]):
        for k in range(result.shape[1]):
            for l in range(result.shape[2]):
                result[m, k, l] = max(arr[m, k * 2, l * 2], arr[m, k * 2, l * 2 + 1],
                                      arr[m, k * 2 + 1, l * 2], arr[m, k * 2 + 1, l * 2 + 1])
    return result


def softmax(arr):
    result = np.zeros(arr.shape, dtype=np.float32)
    for x in range(result.shape[1]):
        for y in range(result.shape[2]):
            s = 0
            for k in range(result.shape[0]):
                s += math.exp(arr[k, x, y])
            for m in range(result.shape[0]):
                result[m, x, y] = math.exp(arr[m, x, y]) / s
    return result


nfilters = 2
kernels = np.random.uniform(-1, 1, size=(nfilters, 3, 3, 3))

img = cv2.imread("lenna.png")

arr = conv(img, kernels, 1)
print("------", "CONV", arr.shape, arr, sep="\n")
arr = normalize(arr, 1, 0)
print("------", "NORMALIZE", arr.shape, arr, sep="\n")
arr = relu(arr)
print("------", "RELU", arr.shape, arr, sep="\n")
arr = maxpooling(arr)
print("------", "MAXPOOLING", arr.shape, arr, sep="\n")
arr = softmax(arr)
print("------", "SOFTMAX", arr.shape, arr, sep="\n")
