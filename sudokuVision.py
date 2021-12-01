#!/bin/python3
import cv2 as cv
import numpy as np
import sys
import getopt
import os

### GLOBAL CONSTANTS ###
# enum for sides
TOP   = 0
BOT   = 1
LEFT  = 2
RIGHT = 3
# for modes
CLASSIC = "classic"
JIGSAW  = "jigsaw"
# counter for correctly predicted solutions
CORRECT = 0
# observable means of numbered cells were surely greater than 15
# observable means of empty cells with noise (after erosion) were less than 1
# we pick a safety net (> 15) just in case
NUMBER_CELL_MIN_MEAN = 15

### FLAGS ###
VERBOSE = False
CHECK   = True
MODE    = CLASSIC

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
    """
    Writes solution to the file passed as argument

    Parameters:
    cells      : matrix of the predicted cells
    outputPath : path towards the output file
    components : separate matrix denoting component (for jigsaw, default is None)

    Returns:
    list : final predicted values, exactly how they appear in the solution file
    """
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

def getSides(cell):
    """
    Checks cell's sides and marks existing sides

    Parameters:
    cell: the cell in question

    Returns:
    bool dict: a dictionary of 4 elements containing True/False
               corresponding to whether or not the cell has that side
               keys -> (TOP, RIGHT, BOT, LEFT)
    """

    cellHeight, cellWidth = np.shape(cell)
    
    sides = {
        TOP   : False,
        RIGHT : False,
        BOT   : False,
        LEFT  : False
    }

    leftSideSum = 0
    rightSideSum = 0
    topSideSum = 0
    botSideSum = 0

    # basically how many lines to check
    # we divide the width / height to this variable to find the nth part of the cell
    nthPartDiv = 8

    # the thick contours will remain in the cells once extracted
    # we can use this to determine the thick contours positions relative to the cell
    # we sum up the pixels that are a part of the contour to determine if there's a contour on that side
    for k in range(cellHeight):
        leftSideSum += np.sum(cell[k][:cellWidth // nthPartDiv]) // 255
        rightSideSum += np.sum(cell[k][cellWidth - cellWidth // nthPartDiv - 1:]) // 255

    for k in range(cellHeight // nthPartDiv):
        topSideSum += np.sum(cell[k]) // 255
        botSideSum += np.sum(cell[cellHeight - k - 1]) // 255
    sides[LEFT]  = leftSideSum  > cellHeight
    sides[TOP]   = topSideSum   > cellWidth
    sides[RIGHT] = rightSideSum > cellWidth
    sides[BOT]   = botSideSum   > cellHeight
 
    return sides
    
def isInBounds(position):
    return 0 <= position[0] and position[0] < 9 and 0 <= position[1] and position[1] < 9

def isValidCell(fromCellSides, toCellSides, direction):
    """
    Function decides whether the cell we're travelling to is valid relative to the direction of travel.
    
    Parameters:
    fromCellSides : tuple returned by getSides() for the cell we're travelling from 
    toCellSides   : tuple returned by getSides() for the cell we're going to travel to
    direction     : int representing the direction of travel

    Returns:
    bool: True if valid cell, False if invalid cell
    """
    # the ints for direction are specially picked so when inverted this wait
    # they will have the corresponding value to the opposed direction
    sideToCheck = direction ^ 1

    # cell is valid only if it doesn't have a contour on that side, so our flood can continue to it
    # we also check if our current cell has a contour on that side
    return not toCellSides[sideToCheck] and not fromCellSides[direction]

       

def markComponents(contourCells):
    """
    Generates a matrix of what component each cell is a part of

    Parameters:
    contourCells: matrix of the cells with only the highlighted thick contour

    Returns:
    np.array: matrix where each cell is marked with a component number
    """
    sidesMatrix = [[{} for _ in range(9)] for _ in range(9)]
    #print("[TOP, RIGHT, BOT, LEFT]")
    for i in range(9):
        for j in range(9):
            sidesMatrix[i][j] =  getSides(contourCells[i][j])
            #print(f"[{int(sidesMatrix[i][j][TOP])},{int(sidesMatrix[i][j][RIGHT])},{int(sidesMatrix[i][j][BOT])},{int(sidesMatrix[i][j][LEFT])}]", end = " ")
        #print()
    currentComponent = 1
    components = np.zeros((9, 9), np.uint8)
    directions = [(-1, 0, TOP), (0, 1, RIGHT), (1, 0, BOT), (0, -1, LEFT)]
    
    # basically breadth-first search
    for i in range(9):
        for j in range(9):
            if components[i][j] == 0:
                queue = [(i,j)]
                components[i][j] = currentComponent
                while len(queue):
                    currPos = queue[0]
                    queue.pop(0)
                    for direction in directions:
                        literalDir = direction[2]
                        newPos = (currPos[0] + direction[0], currPos[1] + direction[1])
                        # if new pos is not in bounds or it's already visited
                        if not isInBounds(newPos) or components[newPos[0]][newPos[1]] != 0:
                            continue
                        
                        # if we can travel in that direction
                        if isValidCell(sidesMatrix[currPos[0]][currPos[1]], sidesMatrix[newPos[0]][newPos[1]], literalDir):                      
                            queue.append(newPos)
                            components[newPos[0]][newPos[1]] = currentComponent
                currentComponent += 1
    return components

def isCorrectlyPredicted(predictedCells, truthPath):
    """
    Checks predicted solution against real solution.

    Parameters:
    predictedCells : matrix of the predicted sudoku puzzle configuration
    truthPath      : path to the true sudoku puzzle configuration

    Returns:
    bool: True if correct, False if incorrect
    """
    global CORRECT

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
    
    CORRECT += 1
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
    # very important: we use the unaltered original image
    warpedImg = cv.warpPerspective(img, perspective, (widthWarp, heightWarp))

    # finally threshold and invert the warped image
    # large blocksize gets rid of more noise
    # now we finally have the sudoku puzzle only
    warpedImg = 255 - cv.adaptiveThreshold(warpedImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 1)
    
    printInfo("Extracting cells...")
    
    # extract the cells
    cells = extractCells(warpedImg)
    # if mode is JIGSAW we also want to extract the cells
    # after highlighting the components contour
    eroded = None
    if MODE == JIGSAW:
        kernel = np.ones((5, 1), np.uint8)
        # erode vertical lines then horizontal lines
        eroded = cv.erode(warpedImg, kernel)
        eroded = cv.erode(eroded, np.transpose(kernel))
        
        # open up
        opened = cv.dilate(eroded, np.ones((3, 3), np.uint8))
        # extract the contour cells
        contourCells = extractCells(opened)
        # based on the contour cells, generate the matrix with components
        componentMatrix = markComponents(contourCells)


    printInfo(f"Writing to file {outputPath}...")
    predictedValues = writeSolution(cells, outputPath, componentMatrix if MODE != CLASSIC else None)
    
    if truthPath:
        print("Correct!" if isCorrectlyPredicted(predictedValues, truthPath) else "Incorrect!")
    
    printInfo("Done!")

def printHelp():
    helpString = ""
    helpString += "Usage: sudokuVision [OPTION]... FILE \n"
    helpString += "Extract sudoku and output data in a file\n"
    helpString += "\n"
    helpString += "Options:\n"
    helpString += "  -h, --help                Display this help and exit\n"
    helpString += "  -v, --verbose             Print more info on the steps\n"
    helpString += "  -c, --check               Check solution against ground truths and count correct guesses\n"
    helpString += "\n"
    helpString += "  -m, --mode=<MODE>         Mode to run on [classic, jigsaw]\n"
    helpString += "  -t, --truth-path=<PATH>   Path to truth files\n"
    helpString += "                            (if not provided, it looks in the same path as the files to run on)\n"
    helpString += "  -f <FILES>                Path to the file(s) to run through Sudoku Vision\n"
    
    helpString += "\n"
    helpString += "Written by Costinteo.\n"
    helpString += "Licensed under GPL v3"
    print(helpString)

if __name__ == "__main__":
    cliOptions, cliArgs = getopt.gnu_getopt(sys.argv[1:], "hvcm:t:f:", ["help", "verbose", "check", "mode=", "truth-path=", "input"])
    print(cliOptions, cliArgs)
    
    paths = cliArgs
    
    for flag, arg in cliOptions:
        print(flag, arg)
        if flag == "-h" or flag == "--help":
            printHelp()
            exit(0)
        if flag == "-m" or flag == "--mode":
            arg = arg.lower()
            if arg == "classic" or arg == "clasic":
                MODE = CLASSIC
            elif arg == "jigsaw" or arg == "jig":
                MODE = JIGSAW
            else:
                print("Wrong mode! Check --help")

    for path in paths:
        # extract filename (using os.sep so it works on any platform)
        filename = path[path.rfind(os.sep) + 1 : path.rfind(".")]
        cwd = path[:path.rfind(os.sep)]
        sudokuVision(path, f".{os.sep}output{os.sep}{filename}_predicted.txt", f"{cwd}{os.sep}{filename}_gt.txt")
    if CHECK:
        print(f"{CORRECT}/{len(paths)} correctly guessed.")
