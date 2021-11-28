#!/bin/python3
from typing import final
import cv2 as cv
import numpy as np

# observable means of numbered cells were surely greater than 10
# observable means of empty cells with noise (after erosion) were less than 1
# we pick a safety net (> 10) just in case
NUMBER_CELL_MIN_MEAN = 10

# return an image preprocessed
def preprocessed(img):
    # getting rid of noise
    blur = cv.GaussianBlur(img.copy(), (9, 9), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    inverted = cv.bitwise_not(thresh)
    
    # dilate after Gaussian thresholding
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv.dilate(inverted, kernel)
    return dilated

# function returns largest contour from an image
def getLargestContour(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    return contours[0]

def euclideanDistance(pointA, pointB):
    return np.sqrt(((pointA[0] - pointB[0]) ** 2) + ((pointA[1] - pointB[1]) ** 2))

def getCornersOrdered(cornerList):
    """
    Matches corners to a predictable clockwise order.

    Order of the returned corners is the following:
    index 0: top left corner
    index 1: top right corner
    index 2: bot left corner
    index 3: bot right corner

    Parameters:
    cornerList: list of corners coordinates

    Returns:
    list: List of the orderedCorners in the order specified
    """
    global imgHeight, imgWidth
    
    orderedCorners = [0, 0, 0, 0]
    for corner in cornerList:
        x = corner[0]
        y = corner[1]

        # returns 1 if corner is right side
        isRight = x > imgWidth // 2
        # returns 1 if corner is bottom side
        isBottom = y > imgHeight // 2 
        
        # using the above booleans, we can determine de index
        orderedCorners[isRight + 2 * isBottom] = corner
    return orderedCorners

def isCellEmpty(cell):
    """
    Checks if cell given as argument is empty.

    Parameters:
    cell: the np.array representing the cell

    Returns:
    bool: True if cell is empty, False otherwise
    """
    cellHeight, cellWidth = np.shape(cell)

    # cut edges of cell to remove possible borders
    # we basically cut 2 fourths on all sides
    cutHeight = cellHeight // 4
    cutWidth = cellWidth // 4
    
    
    cutCell = cell[cutHeight:cellHeight - cutHeight]
    finalCell = np.zeros((cellHeight - 2 * cutHeight, cellWidth - 2 * cutWidth), np.uint8)
    
    for i, line in enumerate(cutCell):
        finalCell[i] = line[cutWidth:cellWidth - cutWidth]
    
    # erode the resulting cell, to make sure we get rid of any remaining noise
    finalCell = cv.erode(finalCell, np.ones((2, 2), np.uint8))

    # now we can accurately determine if a cell is empty or not using its mean
    # we take a safety margin of NUMBER_CELL_MIN_MEAN, just in case there's any noise left
    return np.mean(finalCell) < NUMBER_CELL_MIN_MEAN

def extractCells(img):
    """
    Extracts cells given an image (that is already perfectly cropped to puzzle)

    Parameters:
    img: the image cropped to the puzzle

    Returns:
    np.array: matrix of all the cell representations
    """
    imgHeight = np.shape(img)[0]
    imgWidth = np.shape(img)[1]

    # sudoku puzzle has 81 cells
    # 9 x 9
    cellHeight = imgHeight // 9
    cellWidth = imgWidth // 9

    # 9 x 9 matrix of cells
    cells = [[np.zeros((cellHeight, cellWidth), np.uint8) for _ in range (9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            cell = np.zeros((cellHeight, cellWidth), np.uint8)
            for k in range(cellHeight):
                cell[k] = img[cellHeight * i + k][cellWidth * j: cellWidth * (j + 1)]
            cells[i][j] = cell

    return cells

# function that shows image
def showImg(img, title="default", timer=0):
    cv.imshow(title, img)
    cv.waitKey(timer)
    cv.destroyAllWindows()       

path = "./antrenare/clasic/10.jpg"
colorImg = cv.imread(path)
colorImg = cv.resize(colorImg, (0, 0), fx=0.2, fy=0.2)
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)

imgHeight, imgWidth = np.shape(img)

procImg = preprocessed(img)

# we get the largest contour in procImg, which will be the sudoku puzzle
largestContour = getLargestContour(procImg)
arc = cv.arcLength(largestContour, True)
# poly will contain an approximate polygon based on the largest contour
poly = cv.approxPolyDP(largestContour, 0.015 * arc, True)

corners = []
# we iterate through the poly corners because they are contained in np arrays
for arr in poly:
    corners.append(list(arr[0]))

# we call a function to order the corners clockwise, even if picture is rotated
topLeft, topRight, botLeft, botRight = getCornersOrdered(corners)

# calculate the bottom side and top side using Euclidean distance
# used later to warp perspective
widthBot = euclideanDistance(botLeft, botRight)
widthTop = euclideanDistance(topLeft, topRight)
# we use the maximum distance to determine the width
# of the new warped image
widthWarp = max(int(widthBot), int(widthTop))

# calculate the left side and right side using Euclidean distance
# used later to warp perspective
heightLeft = euclideanDistance(topLeft, botLeft)
heightRight = euclideanDistance(topRight, botRight)
# we use the maximum distance to determine the height
# of the new warped image
heightWarp = max(int(heightLeft), int(heightRight))

newDimensions = np.array([[0, 0], [widthWarp - 1, 0], [widthWarp - 1, heightWarp - 1], [0, heightWarp - 1]], np.float32)
# corners have to be ordered clockwise for the warp
ordCornersArr = np.array([topLeft, topRight, botRight, botLeft], np.float32)

# needed for warped perspective
grid = cv.getPerspectiveTransform(ordCornersArr, newDimensions)
warpedImg = cv.warpPerspective(img, grid, (widthWarp, heightWarp))

# finally threshold and invert the warped image
# now we finally have the sudoku puzzle only
warpedImg = cv.bitwise_not(cv.adaptiveThreshold(warpedImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))

# extract the cells
cells = extractCells(warpedImg)

outputFile = open("output.txt", "w")

# iterate through the cells
# write their values to corresponding output file
for i in range(len(cells)):
    for j in range(len(cells[i])):
        outputFile.write("o" if isCellEmpty(cells[i][j]) else "x")
    outputFile.write("\n")

