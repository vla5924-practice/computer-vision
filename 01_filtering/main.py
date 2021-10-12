import cv2
import numpy

image = cv2.imread("lenna.png")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

cropped = image[faces[0][1]:faces[0][1] + faces[0]
                [3], faces[0][0]:faces[0][0] + faces[0][2]]

edged = cv2.Canny(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 100, 200)
# TODO: Remove edges <10px???

kernel = numpy.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=numpy.uint8)

dilated = cv2.dilate(edged, kernel)

blurred = cv2.GaussianBlur(dilated, (5, 5), 0, 0)
normalized = cv2.normalize(blurred, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

bilateral = cv2.bilateralFilter(cropped, 10, 75, 75)

# TODO: Sharpening???
kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(cropped, -1, kernel)

result = numpy.zeros(cropped.shape, dtype=numpy.uint8)
width, height, channels = result.shape

for x in range(width):
    for y in range(height):
        for c in range(channels):
            result[x, y, c] = normalized[x, y] * sharpened[x, y, c] + \
                (1 - normalized[x, y]) * bilateral[x, y, c]

cv2.imshow('01_filtering', result)
cv2.waitKey(0)
