#!/bin/python3
import cv2 as cv
import numpy as np

# observable means of numbered cells were surely greater than 20
# observable means of empty cells with noise (after erosion) were less than 1
# we pick a safety net (> 20) just in case
NUMBER_CELL_MIN_MEAN = 20

VERBOSE = False

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

def getCornersOrdered(cornerList, shape):
    """
    Matches corners to a predictable clockwise order.

    Order of the returned corners is the following:
    index 0: top left corner
    index 1: top right corner
    index 2: bot left corner
    index 3: bot right corner

    Parameters:
    cornerList: list of corners coordinates
         shape: pair of height, width of the original image

    Returns:
    list: List of the orderedCorners in the order specified
    """
    imgHeight, imgWidth = shape

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

def isCorrectlyPredicted(predictedCells, truthPath):
    """
    Checks predicted solution against real solution.

    Parameters:
    predictedCells: matrix of the predicted sudoku puzzle configuration
         truthPath: path to the true sudoku puzzle configuration

    Returns:
    bool: True if correct, False if incorrect
    """
    truthCells = open(truthPath, "r").readlines()
    for i in range(len(predictedCells)):
        for j in range(len(predictedCells[i])):
            if predictedCells[i][j] != ord(truthCells[i][j]):
                if VERBOSE:
                    print(f"Solution failed at {i}, {j}")
                    print(f"Predicted: {chr(predictedCells[i][j])}, {predictedCells[i][j]}")
                    print(f"Truth: {truthCells[i][j]}, {ord(truthCells[i][j])}")
                return False
    return True

# function that shows image
def showImg(img, title="default", timer=0):
    cv.imshow(title, img)
    cv.waitKey(timer)
    cv.destroyAllWindows()       

def sudokuVision(inputPath, outputPath, truthPath=None):
    if VERBOSE:
        print(f"Running {inputPath} through Sudoku Vision...")
    
    try:
        img = cv.imread(inputPath, cv.IMREAD_GRAYSCALE)
    except:
        print("Unexpected error occured. Make sure you passed a path to an image as argument!")
    
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)

    if VERBOSE:
        print("Preprocessing image...")
    procImg = preprocessed(img)

    if VERBOSE:
        print("Approximating polygon...")
    # we get the largest contour in procImg, which will be the sudoku puzzle
    largestContour = getLargestContour(procImg)
    arc = cv.arcLength(largestContour, True)
    # poly will contain an approximate polygon based on the largest contour
    poly = cv.approxPolyDP(largestContour, 0.015 * arc, True)

    corners = []
    # we iterate through the poly corners because they are contained in np arrays
    for arr in poly:
        corners.append(list(arr[0]))
    
    if VERBOSE:
        print("Grabbing corners...")
    # we call a function to order the corners clockwise, even if picture is rotated
    topLeft, topRight, botLeft, botRight = getCornersOrdered(corners, np.shape(procImg))

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

    if VERBOSE:
        print("Cropping to sudoku puzzle...")
    # needed for warped perspective
    grid = cv.getPerspectiveTransform(ordCornersArr, newDimensions)
    warpedImg = cv.warpPerspective(img, grid, (widthWarp, heightWarp))

    # finally threshold and invert the warped image
    # now we finally have the sudoku puzzle only
    warpedImg = cv.bitwise_not(cv.adaptiveThreshold(warpedImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))

    if VERBOSE:
        print("Extracting cells...")
    # extract the cells
    cells = extractCells(warpedImg)

    if VERBOSE:
        print(f"Writing to file {outputPath}...")
    outputFile = open(outputPath, "w")
    predictedValues = np.empty((9, 9), np.uint8)
    # iterate through the cells
    # write their values to corresponding output file
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            outputFile.write("o" if isCellEmpty(cells[i][j]) else "x")
            predictedValues[i][j] = ord("o") if isCellEmpty(cells[i][j]) else ord("x")
        outputFile.write("\n")
    
    if truthPath:
        print("Correct!" if isCorrectlyPredicted(predictedValues, truthPath) else "Incorrect!")
    if VERBOSE:
        print("Done!")


if __name__ == "__main__":
    #VERBOSE = True
    sudokuVision("./antrenare/clasic/01.jpg", "./output.txt", "./antrenare/clasic/01_gt.txt")
