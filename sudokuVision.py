#!/bin/python3

import sys
import cv2 as cv
import getopt
import numpy as np

MAX_PIXEL_VALUE = 255

LINE_KERNEL_SIZE = 100
VERTICAL_KERNEL = np.transpose(np.array([LINE_KERNEL_SIZE*[0],LINE_KERNEL_SIZE*[1],LINE_KERNEL_SIZE*[0]])) / LINE_KERNEL_SIZE
HORIZONTAL_KERNEL = np.array([LINE_KERNEL_SIZE*[0],LINE_KERNEL_SIZE*[1],LINE_KERNEL_SIZE*[0]]) / LINE_KERNEL_SIZE

def normalizeImage(img):
    noise = cv.dilate(img, np.ones((7,7), np.uint8))
    showImg(noise)
    blur = cv.medianBlur(noise, 21)
    showImg(blur)
    res = MAX_PIXEL_VALUE - cv.absdiff(img, blur)
    showImg(res)
    noShadow = cv.normalize(res, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    return noShadow

def filter(img, kernel):
    filtered = cv.filter2D(img, -1, kernel)
    filtered[filtered > 0] = MAX_PIXEL_VALUE
    return filtered

def extractLines(filteredImg, axis):
    # if axis passed as arg is "vertical"   -> axisId is 0
    # if axis passed as arg is "horizontal" -> axisId is 1
    # this is done to avoid apparent magic numbers at function call
    axisId = 0 if axis.lower() == "vertical" else 1

    mask = (MAX_PIXEL_VALUE - filteredImg) // MAX_PIXEL_VALUE
    mask = np.sum(mask, axis = axisId)
    lines = []
    maxLineWidth = 0
    currentLineWidth = 0
    print(mask)
    for i in range(0, len(mask)):
        if mask[i] != 0:
            lineTuple = (i, 0) if axisId == 0 else (0, i)
            shapeTuple = (i, filteredImg.shape[axisId]) if axisId == 0 else (filteredImg.shape[axisId], i)
            currentLineWidth += 1
            lines.append([lineTuple, shapeTuple])
        elif maxLineWidth < currentLineWidth:
            maxLineWidth = currentLineWidth 
            currentLineWidth = 0

    distinctLines = [lines[0]]
    print(maxLineWidth)
    for line in lines:
       if line[0][axisId] - distinctLines[-1][0][axisId] > maxLineWidth:
           distinctLines.append(line)
    return distinctLines



def showImg(img, title="default", timer=0):
    cv.imshow(title, img)
    cv.waitKey(timer)
    cv.destroyAllWindows()


img = cv.imread("antrenare/clasic/01.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (0,0), fx=0.2, fy=0.2) 
showImg(img)
img = normalizeImage(img)
showImg(img)
mean = np.mean(img)
_, thresh = cv.threshold(img, mean, MAX_PIXEL_VALUE, cv.THRESH_BINARY)
showImg(thresh)
linesVertical = filter(thresh, VERTICAL_KERNEL)
linesHorizontal = filter(thresh, HORIZONTAL_KERNEL)
showImg(linesVertical)
showImg(linesHorizontal)
for line in extractLines(linesHorizontal, "horizontal"):
    print(line)
    cv.line(img, line[0], line[1], (0, 0, 255), 1)

for line in extractLines(linesVertical, "vertical"):
    cv.line(img, line[0], line[1], (0, 0, 255), 1)
    print(line)
showImg(img)
#opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

print("TEST")

