import numpy as np
import cv2
import math
from enum import Enum

PAPER_WIDTH = 3 #in
PAPER_HEIGHT = 5 #in
DRAWN_HEIGHT = 4.5 #in
CAM_HEIGHT = 1280.0 #px
CAM_WIDTH = 720.0 #px
MM_PER_IN = 25.4 #you probably can keep this as is...
DRAWN_WIDTH = CAM_WIDTH/CAM_HEIGHT*DRAWN_HEIGHT #in
SCALAR = DRAWN_HEIGHT/CAM_HEIGHT*MM_PER_IN #mm/pixel
OFFSETY = (PAPER_HEIGHT-DRAWN_HEIGHT)/2.0*MM_PER_IN #mm
OFFSETX = (PAPER_WIDTH-DRAWN_WIDTH)/2.0*MM_PER_IN #mm

Z_RETRACT_HEIGHT = 1.0 #height to raise z axis for moves
Z_SPEED_MULT = 1.0/8.0 #relative speed of z axis compared to x.y

GCODE_FILE_NAME = "lines.gcode" #where gcode gets saved

class Line:
    '''
    Simple struct to hold start and end points of a line
    '''
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def nearestLine(line, lines):
    '''
    Finds the nearest line to "line" in the list of lines and returns it.
    If the list of lines is empty, None is returned.
    '''
    if line is None or lines is None:
        return None
    if len(lines) < 1:
        return None
    minDist = 999999999
    bestLine = lines[0]
    for ln in lines:
        if ln.x1-line.x2 > minDist:
            continue
        if ln.y1-line.y2 > minDist:
            continue
        dist = pow(ln.x1-line.x2,2)+pow(ln.y1-line.y2,2)
        if dist < minDist and line is not ln:
            minDist = dist
            bestLine = ln
    return (bestLine, minDist)

def find_lines(input):
    """Finds lines in a single channel image.
    Args:
        input: A numpy.ndarray.
    Returns:
        A filtered list of Lines.
    """
    detector = cv2.createLineSegmentDetector(1)
    if (len(input.shape) == 2 or input.shape[2] == 1):
        lines = detector.detect(input)
    else:
        tmp = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        lines = detector.detect(tmp)
    output = []
    if len(lines) != 0 and lines[0] is not None:
        for i in range(1, len(lines[0])):
            tmp = Line(lines[0][i, 0][0], lines[0][i, 0][1],
                            lines[0][i, 0][2], lines[0][i, 0][3])
            output.append(tmp)
    return output

gcodeFile = None
def printGcode(cmd, lastLine=False):
    '''
    Prints the input line (cmd) of gcode to the console and saves it to a file.
    If lastLine is true then the file is closed.
    '''
    global gcodeFile
    if gcodeFile is None:
        gcodeFile=open(GCODE_FILE_NAME, "w+")
    gcodeFile.write(cmd + "\r\n")
    print(cmd)
    if lastLine:
        gcodeFile.close()
        gcodeFile = None

def printFrame(lastLines):
    '''
    Takes in a list of lines and generates the gcode to plot them.
    '''
    print(len(lastLines))
    linesIter = lastLines.copy()
    printImg = np.zeros((int(CAM_WIDTH),int(CAM_HEIGHT),3), np.uint8)
    printImg[:,:] = (255,255,255)
    for line in linesIter:
        cv2.line(printImg,(line.x1,line.y1),(line.x2,line.y2),(0,0,255),1)
    cv2.putText(printImg,"Printing...", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    cv2.imshow('frame',printImg)

    lastX = 0
    lastY = 0
    printGcode("G21") #millimeters
    printGcode("G90")
    printGcode("G0 Z5 F500")
    printGcode("G28 X")
    printGcode("G28 Y")
    distToLine = np.sqrt(pow(line.x1-lastX,2)+pow(line.y1-lastY,2))
    if distToLine < 3:
        printGcode("G0 Z0 F1000")
    line = linesIter[0]
    counter = 0
    percentage = 0
    lastPercentage = 0
    success = True
    while line is not None:
        counter+=1
        distToLine = np.sqrt(pow(line.x1-lastX,2)+pow(line.y1-lastY,2))
        if distToLine > 3:
            startX = lastX
            startY = lastY
            finalY = (float(line.x1)*SCALAR)+OFFSETY
            finalX = (float(line.y1)*SCALAR)+OFFSETX
            dist_travel = np.sqrt(pow(finalX-startX,2)+pow(finalY-startY,2))
            if (dist_travel * Z_SPEED_MULT < Z_RETRACT_HEIGHT): #triange retract
                zMax = dist_travel * Z_SPEED_MULT
                midX = (finalX+startX)/2.0
                midY = (finalY+startY)/2.0
                printGcode(f"G0 X{midX:.3f} Y{midY:.3f} Z{zMax:.3f} F12000")
                printGcode(f"G0 X{finalX:.3f} Y{finalY:.3f} Z0.000 F12000")
            else: #trapeziodal retract
                travel_until_retracted = Z_RETRACT_HEIGHT / dist_travel * Z_SPEED_MULT
                print(f"start: {startX:.2f} end:{finalX:.2f} trav:{travel_until_retracted:.3f}")
                mid1X = (finalX-startX)*travel_until_retracted+startX
                mid2X = finalX-(finalX-startX)*travel_until_retracted
                mid1Y = (finalY-startY)*travel_until_retracted+startY
                mid2Y = finalY-(finalY-startY)*travel_until_retracted
                printGcode(f"G0 X{mid1X:.3f} Y{mid1Y:.3f} Z{Z_RETRACT_HEIGHT:.3f} F12000")
                printGcode(f"G0 X{mid2X:.3f} Y{mid2Y:.3f} Z{Z_RETRACT_HEIGHT:.3f} F12000")
                printGcode(f"G0 X{finalX:.3f} Y{finalY:.3f} Z0.000 F12000")
        printGcode(f"G1 X{(float(line.y2)*SCALAR)+OFFSETX:.3f} Y{(float(line.x2)*SCALAR)+OFFSETY:.3f} F6000")
        cv2.line(printImg,(line.x1,line.y1),(line.x2,line.y2),(0,255,0),1)
        if counter % 10 == 0:
            percentage = 100-int(99*len(linesIter)/len(lastLines))
            cv2.putText(printImg,f"Printing... {lastPercentage}%", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
            cv2.putText(printImg,f"Printing... {percentage}%", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
            lastPercentage = percentage
            cv2.imshow('frame',printImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                printGcode("G0 Z5", True)
                success = False
                break
        lastX = float(line.y2)*SCALAR+OFFSETX
        lastY = float(line.x2)*SCALAR+OFFSETY
        if len(linesIter) > 0:
            (nextLine, minDist) = nearestLine(line, linesIter)
            linesIter.remove(line)
            line = nextLine
            continue
        line = None
        break
    if success == True:
        printGcode("G0 Z5 F1000")
        printGcode(f"G0 X{OFFSETX:.3f} Y{OFFSETY:.3f} F8000")
        printGcode("G0 Z0 F1000")
        printGcode(f"G1 X{(CAM_WIDTH*SCALAR)+OFFSETX:.3f} Y{OFFSETY:.3f} F4000")
        printGcode(f"G1 X{(CAM_WIDTH*SCALAR)+OFFSETX:.3f} Y{(CAM_HEIGHT*SCALAR)+OFFSETY:.3f} F4000")
        printGcode(f"G1 X{OFFSETX:.3f} Y{(CAM_HEIGHT*SCALAR)+OFFSETY:.3f} F4000")
        printGcode(f"G1 X{OFFSETX:.3f} Y{OFFSETY:.3f} F8000")
        printGcode("G0 Z5 F1000", True)
    percentage = 100
    cv2.putText(printImg,f"Printing... {lastPercentage}%", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    cv2.putText(printImg,f"Printing... {percentage}%", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
    cv2.imshow('frame',printImg)
    cv2.waitKey(5000)



# -----Program code-----

cap = cv2.VideoCapture(0)
img = np.zeros((int(CAM_WIDTH),int(CAM_HEIGHT),3), np.uint8)
state = 0
while(True):
    key = cv2.waitKey(1)
    _, frame = cap.read()

    if state == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img[:,:] = (255,255,255)
        h,s,v = cv2.split(hsv)
        lines = find_lines(v)
        lines.extend(find_lines(s))
        for line in lines:
            cv2.line(img,(line.x1,line.y1),(line.x2,line.y2),(255,0,0),1)
        cv2.putText(img,"Press C to capture image", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        cv2.imshow('frame',img)
        lastLines = lines.copy()
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            state = 1

    elif state == 1:
        img[:,:] = (255,255,255)
        for line in lastLines:
            cv2.line(img,(line.x1,line.y1),(line.x2,line.y2),(255,0,0),1)
        cv2.putText(img,"Press C to redo, P to print", (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
        cv2.imshow('frame',img)
        state = 2

    elif state == 2:
        if key & 0xFF == ord('c'):
            state = 0
        elif key & 0xFF == ord('p'):
            print("Printing photo")
            printFrame(lastLines)
            state = 0

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
