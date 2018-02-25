# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import time
import sys
# Global setting
'''
if(len(sys.argv) < 2):
    print('Missing argument: filename')
    sys.exit()
'''

videoname = 'barriers.avi' #sys.argv[1]
cap = cv2.VideoCapture(videoname)
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
totalDelta = 0
k = cv2.waitKey(1) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
    sys.exit()
# Callback functions
def brighten(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    lim = 255 - delta
    v[v > lim] = 255
    v[v <= lim] += delta
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def darken(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    lim = 0 + delta
    v[v < lim] = 0
    v[v >= lim] -= delta
    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def edit(x):
    global frame, totalDelta
    delta = cv2.getTrackbarPos('Delta','Curve Tool')
    b = cv2.getTrackbarPos('Brighten','Curve Tool')
    d = cv2.getTrackbarPos('Darken', 'Curve Tool')
    if(b==1):
        frame = brighten(frame, delta)
        totalDelta += delta
    if(d==1):
        frame = darken(frame, delta)
        totalDelta += delta
    time.sleep(1)
    cv2.setTrackbarPos('Apply settings', 'Curve Tool', 0)

def plot(x):
    hist, bins = np.histogram(frame.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(frame.flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    cv2.setTrackbarPos('Histogram','Curve Tool',0)
    plt.show()

def nothing(x):
    pass

def done():
    cv2.destroyAllWindows()
    sys.exit()

# Setup user interface
cv2.namedWindow('Curve Tool')
cv2.createTrackbar('Delta', 'Curve Tool', 0, 50, nothing)
cv2.createTrackbar('Brighten', 'Curve Tool', 0, 1, nothing)
cv2.createTrackbar('Darken', 'Curve Tool', 0, 1, nothing)
cv2.createTrackbar('Apply settings', 'Curve Tool', 0, 1, edit)
cv2.createTrackbar('Histogram', 'Curve Tool', 0, 1, plot)
cv2.createTrackbar('Save', 'Curve Tool', 0 ,1 ,nothing)
cv2.createTrackbar('Close', 'Curve Tool', 0 ,1 ,nothing)

# The program loop
while(cv2.getTrackbarPos('Close','Curve Tool') == 0):
    if(ret == True):
        cv2.imshow('Curve Tool', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if(cv2.getTrackbarPos('Save', 'Curve Tool') == 1):
        # Edit and Save the whole video
        while(ret):
            if(totalDelta > 0):
                frame = brighten(frame, totalDelta)
            if(totalDelta < 0):
                frame = darken(frame, (-1)*totalDelta)
            out.write(frame)
            ret, frame = cap.read()
            if(ret == True):
                cv2.imshow('Curve Tool', frame)
            else:
                done()