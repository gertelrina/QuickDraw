
import numpy as np
import cv2
from collections import deque
from gtts import gTTS
from pygame import mixer
import time

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=1000)]

bindex = 0

colors = [(255, 255, 255)]

colorIndex = 0

# Setup the Paint interface
paintWindow = np.zeros((1280, 1280,3)) + 0
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
cv2.getWindowImageRect('Paint')

# Load the video
camera = cv2.VideoCapture(0)

# Switcher for 1.TurnOn/Off drawing 2.Once append deq 
swTurnOnOff = 1
onceAppend = 1

# Counter of image
counterIMG = 0

# String of prediction type and label
predictString = ' '

# Language of voice
languageVo = 'en'

# For vocalize prediction
mixer.init()

# File for save and load vocalize
fileVoice = "1.mp3"

while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.putText(frame, "CLEAR ALL - press the 'c' key", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "SENT - press the 's' key", (49, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "STOP DRAWING - press the 'h' key", (49, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "{} sent".format(counterIMG) , (600, 33), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if(swTurnOnOff==1):
        cv2.putText(frame, "Turn On" , (800, 33), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Turn Off" , (800, 33), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, predictString, (49, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Check to see if we have reached the end of the video
    if not grabbed:
        break

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours in the image
    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
        # Sort the contours and find the largest one -- we
        # will assume this contour correspondes to the area
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Append new points in currently deq
        if colorIndex == 0 and swTurnOnOff == 1:
            bpoints[bindex].appendleft(center)

    # Clear paintWindow 
    if cv2.waitKey(20) & 0xFF == ord("c"):
        bpoints = [deque(maxlen=512)]
        bindex = 0
        paintWindow[:,:,:] = 0

    # Sent key
    if (cv2.waitKey(20) & 0xFF == ord("s")):
        # ..... modifying line for project .....
        cv2.imwrite("image_{}.jpg".format(c), paintWindow)
        # ..... modifying line for project .....

        # Change PredictString
        # predictString = "Predictions: " + str(predictions) + str(labels) 

        #  Create and play vocalize of prediction
        voice = gTTS(text = predictString, lang = languageVo, slow = False)
        voice.save(fileVoice)
        mixer.music.load(fileVoice)
        mixer.music.play()

        # Clear paintWindow
        bpoints = [deque(maxlen=512)]
        bindex = 0
        paintWindow[:,:,:] = 0
        counterIMG+=1

    # Change switcher for stop/start drawing
    if cv2.waitKey(20) & 0xFF == ord("h"):
        swTurnOnOff = -swTurnOnOff
        onceAppend = 1

    # Create the next new deq for lines   
    if swTurnOnOff == -1 and onceAppend == 1:
        bpoints.append(deque(maxlen=1000))
        bindex += 1
        onceAppend = -1


    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Draw lines of all the colors
    # points = [bpoints]
    points = [bpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 10)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 10)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    
# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
