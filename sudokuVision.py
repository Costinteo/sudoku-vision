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
    blur = cv.medianBlur(noise, 21)
    res = MAX_PIXEL_VALUE - cv.absdiff(img, blur)
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
    for line in lines:
       if line[0][axisId] - distinctLines[-1][0][axisId] > maxLineWidth:
           distinctLines.append(line)
    return distinctLines



def showImg(img, title="default", timer=0):
    cv.imshow(title, img)
    cv.waitKey(timer)
    cv.destroyAllWindows()


for fn in range(1, 21):
    add_0 = ""
    if fn < 10:
        add_0 = "0"
    path = f"./fake_test/clasic/{add_0}{fn}.jpg"
    print(path)
    colorImg = cv.imread(path)
    colorImg = cv.resize(colorImg, (0,0), fx=0.2, fy=0.2) 
    img = cv.cvtColor(colorImg, cv.COLOR_BGR2GRAY)
    img = normalizeImage(img)
    mean = np.mean(img)
    _, thresh = cv.threshold(img, mean, MAX_PIXEL_VALUE, cv.THRESH_BINARY)
    showImg(thresh)
    filteredLinesVertical = filter(thresh, VERTICAL_KERNEL)
    filteredLinesHorizontal = filter(thresh, HORIZONTAL_KERNEL)
    linesHorizontal = extractLines(filteredLinesHorizontal, "horizontal")
    linesVertical = extractLines(filteredLinesVertical, "vertical")
    for line in linesHorizontal:
        cv.line(colorImg, line[0], line[1], (0, 0, 255), 2)
        
    for line in linesVertical:
        cv.line(colorImg, line[0], line[1], (0, 0, 255), 2)
    
    showImg(colorImg)
    
    outputPath = f"./fisiere_solutie/Grigore_Costin_333/clasic/{add_0}{fn}"
    output = open(outputPath, "w")
    
    for i in range(len(linesHorizontal) - 1):
        firstH = linesHorizontal[i][0][1]
        secondH = linesHorizontal[i + 1][0][1]
        for j in range(len(linesVertical) - 1):
            firstV = linesVertical[j][0][0]
            secondV = linesVertical[j + 1][0][0]
            sq = np.zeros((abs(secondH - firstH), abs(secondV - firstV)), np.uint8)
            for nr, line in enumerate(img[firstH:secondH]):
                sq[nr] = line[firstV:secondV]
            #showImg(sq[25:((np.shape(sq)[1]) - 15):])
            smallSquare = np.zeros((25,25), np.uint8)
            for k in range(25):
                smallSquare[k] = sq[k + 20][20:45]
            #print(np.mean(smallSquare))
            #showImg(smallSquare)
            smallSquare[smallSquare >= 150] = 255
            smallSquare[smallSquare < 150] = 0
            if np.mean(smallSquare) == 255:
                output.write("o")
            else:
                output.write("x")
        if i < len(linesHorizontal) - 2:
            output.write("\n")
    

