#!/bin/python3
import cv2 as cv
import numpy as np
import sys
import getopt
import os

# observable means of numbered cells were surely greater than 15
# observable means of empty cells with noise (after erosion) were less than 1
# we pick a safety net (> 15) just in case
NUMBER_CELL_MIN_MEAN = 15

VERBOSE = False
CLASSIC = "classic"
JIGSAW = "jigsaw"
MODE = JIGSAW

def printInfo(infoString):
    if not VERBOSE:
        return
    print(infoString)

# return a preprocessed image
def preprocessed(img):
    # getting rid of noise
    # using Gausian with 9x9 kernel
    blur = cv.GaussianBlur(img.copy(), (9, 9), 0)
    # adaptive thresholding of blocksize 11
    thresh = 255 - cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    # dilate after Gaussian thresholding
    # we use a cross-shaped kernel to not dilate more than necessary
    # I've tried a square shaped kernel and it left more noise than I'd like
    crossKernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # squareKernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(thresh, crossKernel)
    
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
    
    # apply Opening on the final cell one more time to get rid of ANY possible noise
    # we SHOULD be left with the digit only
    # hopefully this isn't overkill
    finalCell = cv.morphologyEx(finalCell, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
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
                cell[k] = img[cellHeight * i + k][cellWidth * j : cellWidth * (j + 1)]
            cells[i][j] = cell
    return cells

def writeSolution(cells, outputPath, components = None):
    outputFile = open(outputPath, "w")
    predictedValues = [["" for _ in range(9)] for _ in range(9) ]
    # iterate through the cells
    # write their values to corresponding output file
    for i in range(9):
        for j in range(9):
            charToWrite = "o" if isCellEmpty(cells[i][j]) else "x"
            if type(components) is np.ndarray:
                charToWrite = str(components[i][j]) + charToWrite
            outputFile.write(charToWrite)
            predictedValues[i][j] = charToWrite
        if i != 8:
            outputFile.write("\n")
    return predictedValues

def isNewComponent(cell, y, x):
    """
    Decides if cell passed as arg is a part of a new component or not

    Parameters:
    cell: the cell in question
       y: the y position in the matrix
       x: the x position in the matrix

    Returns:
    int: 0 if it's a new component
         1 if it's a part of the left one
         2 if it's a part of the above one
         3 if it's a part of both left and above
    """

    cellHeight, cellWidth = np.shape(cell)

    leftSideSum = 0
    topSideSum = 0

    noLeft = False
    noTop = False
    
    # the thick contours will remain in the cells once extracted
    # we can use this to determine the thick contours positions relative to the cell
    # we sum up the pixels that are a part of the contour to determine if there's a contour
    # then we use this to determine if the cell is part of a new component or not
    for k in range(cellHeight):
        leftSideSum += np.sum(cell[k][:cellWidth // 7]) // 255

    for k in range(cellHeight // 7):
        topSideSum += np.sum(cell[k]) // 255

    if leftSideSum < cellHeight and x > 0:
        noLeft = True

    if topSideSum < cellWidth and y > 0:
        noTop = True

    if noLeft and noTop:
        return 3
    elif noLeft:
        return 1
    elif noTop:
        return 2

    return 0
    

def markComponents(contourCells):
    """
    Generates a matrix of what component each cell is a part of

    Parameters:
    contourCells: matrix of the cells with only the highlighted thick contour

    Returns:
    np.array: matrix where each cell is marked with a component number
    """
    
    # this doesn't work!!!
    # I need LEE to do it
    # FUCK THIS
    # to do:
    # lee using god damn code matrix

    # current component number
    compNumber = 1
    components = np.zeros((9, 9), np.uint8) 
    showImg(255 - contourCells[5][8])
    for i in range(9):
        print()
        for j in range(9):
            code = isNewComponent(contourCells[i][j], i, j)
            print(code, end = " ")
            # new component
            if code == 0:
                components[i][j] = compNumber
                compNumber += 1
            # component same as left
            elif code == 1:
                components[i][j] = components[i][j-1]
            # component same as above
            elif code == 2:
                components[i][j] = components[i-1][j]
            # component is interior or in need of flood
            elif code == 3:
                # first we check if flood is needed
                # by verifying if the leftside cell is different from topside component
                if components[i][j-1] != components[i-1][j]:
                    # we flood to overwrite any cells we mistook
                    # for a new component
                    newCompValue = min(components[i-1][j], components[i][j-1])
                    compToFlood = max(components[i-1][j], components[i][j-1])
    
                
                    if components[i-1][j] > components[i][j-1]:
                        # flood upwards only, special case
                        for k in range(i - 1, -1, -1):
                            if components[k][j] != compToFlood:
                                break
                            components[k][j] = newCompValue
                    else:
                        # start flooding horizontally
                        for k in range(j - 1, -1, -1):
                            # start flooding vertically
                            for l in range(i - 1, -1, -1):
                                # if we reached vertical end of current component
                                if components[l][k] != compToFlood:
                                    break
                                components[l][k] = newCompValue
                            # if we reached horizontal end of current component
                            if components[i][k] != compToFlood:
                                #print(i, k, compToFlood, newCompValue)
                                break
                            components[i][k] = newCompValue
                    # if we flooded the last component used
                    # we need to decrement compNumber
                    compNumber -= 1
                # we take the above component in all cases
                components[i][j] = components[i-1][j]
            #showImg(255 - contourCells[i][j])
    print()
    return components

def isCorrectlyPredicted(predictedCells, truthPath):
    """
    Checks predicted solution against real solution.

    Parameters:
    predictedCells: matrix of the predicted sudoku puzzle configuration
         truthPath: path to the true sudoku puzzle configuration

    Returns:
    bool: True if correct, False if incorrect
    """
    try:
        truthFile = open(truthPath, "r")
    except:
        print("Ground truth path incorrect!")
        exit(1)

    truthCells = [["" for _ in range(9)] for _ in range(9)]
    if MODE == CLASSIC:
        truthCells = truthFile.readlines()   
    elif MODE == JIGSAW:
        for i, line in enumerate(truthFile.readlines()):
            for j in range(9):
                truthCells[i][j] = line[j*2] + line[j*2+1]


    for i in range(9):
        for j in range(9):
            if predictedCells[i][j] != truthCells[i][j]:
                if VERBOSE:
                    print(f"Solution failed at {i}, {j}")
                    print(f"Predicted: {predictedCells[i][j]}")
                    print(f"Truth: {truthCells[i][j]}")
                return False
    return True

# function that shows image
def showImg(img, title="default", timer=0):
    cv.imshow(title, img)
    cv.waitKey(timer)
    cv.destroyAllWindows()

def sudokuVision(inputPath, outputPath, truthPath=None):
    print(f"Running {inputPath} through Sudoku Vision...")
    

    img = cv.imread(inputPath, cv.IMREAD_GRAYSCALE)

    try:
        img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    except:
        print("Unexpected error occured. Make sure you passed a path to an image as argument!")
        exit(1)

    printInfo("Preprocessing image...")
    procImg = preprocessed(img)

    printInfo("Approximating polygon...")
    # we get the largest contour in procImg, which will be the sudoku puzzle
    largestContour = getLargestContour(procImg)
    arc = cv.arcLength(largestContour, True)
    # poly will contain an approximate polygon based on the largest contour
    poly = cv.approxPolyDP(largestContour, 0.015 * arc, True)

    corners = []
    # we iterate through the poly corners because they are contained in np arrays
    for arr in poly:
        corners.append(list(arr[0]))
    
    printInfo("Grabbing corners...")
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

    # contains the four corners to which to warp
    newDimensions = np.array([[0, 0], [widthWarp, 0], [widthWarp, heightWarp], [0, heightWarp]], np.float32)
    # corners have to be ordered clockwise (like above) for the warp
    ordCornersArr = np.array([topLeft, topRight, botRight, botLeft], np.float32)

    printInfo("Cropping to sudoku puzzle...")
    # needed for warped perspective
    perspective = cv.getPerspectiveTransform(ordCornersArr, newDimensions)
    warpedImg = cv.warpPerspective(img, perspective, (widthWarp, heightWarp))

    # finally threshold and invert the warped image
    # large blocksize gets rid of more noise
    # now we finally have the sudoku puzzle only
    warpedImg = 255 - cv.adaptiveThreshold(warpedImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 1)
    
    #showImg(255 -warpedImg)

    printInfo("Extracting cells...")
    
    # extract the cells
    cells = extractCells(warpedImg)
    # if mode is JIGSAW we also want to extract the cells
    # after highlighting the components contour
    eroded = None
    if MODE == JIGSAW:
        dilated = cv.dilate(warpedImg, np.ones((2, 2), np.uint8))
        kernel = np.ones((6, 1), np.uint8)
        # erode vertical lines then horizontal lines
        eroded = cv.erode(dilated, kernel)
        eroded = cv.erode(eroded, np.transpose(kernel))
        # extract the contour cells
        contourCells = extractCells(eroded)
        # based on the contour cells, generate the matrix with components
        componentMatrix = markComponents(contourCells)
        print(componentMatrix)
        showImg(eroded)

    printInfo(f"Writing to file {outputPath}...")
    predictedValues = writeSolution(cells, outputPath, componentMatrix if MODE != CLASSIC else None)
    
    if truthPath:
        print("Correct!" if isCorrectlyPredicted(predictedValues, truthPath) else "Incorrect!")
    
    printInfo("Done!")

if __name__ == "__main__":
    paths = sys.argv[1:]
    #VERBOSE = True
    MODE = JIGSAW
    for path in paths:
        # extract filename (using os.sep so it works on any platform)
        filename = path[path.rfind(os.sep) + 1 : path.rfind(".")]
        cwd = path[:path.rfind(os.sep)]
        sudokuVision(path, f".{os.sep}output{os.sep}{filename}_predicted.txt", f"{cwd}{os.sep}{filename}_gt.txt")
