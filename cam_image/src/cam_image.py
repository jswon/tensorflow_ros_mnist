#!/usr/bin/env python

import numpy as np
import cv2
import sys
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

IMAGE_SIZE = 28

# Image Publisher initialize
rospy.init_node('Image_node', anonymous=True)
pub = rospy.Publisher('video',Image,queue_size=1) #topic configuration

cam = cv2.VideoCapture(0)
rate = rospy.Rate(1) # video relay, 1Hz
bridge = CvBridge()

while not rospy.is_shutdown():
	ret, frame = cam.read() # Image Read.

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray Image
	gray = cv2.GaussianBlur(gray,(3,3),0)
	gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,75,10)

	img,cnts,hierarchy = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	a = []

	# ROI Filtering. 
	for c in cnts:
		approx = cv2.approxPolyDP(c, 3, True)
		area = cv2.contourArea(c)

		x,y,w,h = cv2.boundingRect(c)

		if w <20 or h <20:
			continue

		if area < 50 or area >800:
			continue

		ratio = float(h)/float(w) # ratio = heigt/width

		if ratio > 1.7 or ratio <0.6:
			continue

		#Extract ROI
		a.append(c)

	length = range(len(a))

	print(length)

	for k in length:
		x,y,w,h = cv2.boundingRect(a[k])
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		#cropped = cv2.resize(frame[y:y+h,x:x+w],(IMAGE_SIZE,IMAGE_SIZE))

		c_x = x+int(w/2) 
		c_y = y+int(h/2)

		if not c_x-30 <0 and not c_y-30<0 :
			cropped = cv2.resize(gray[c_y-30:c_y+30,c_x-30:c_x+30],(IMAGE_SIZE,IMAGE_SIZE))
			cv2.imshow('%s,cropped image'%(k+1), cropped)
			pub.publish(bridge.cv2_to_imgmsg(cropped)) # Cropped Image topic publishing		i = i+1

	cv2.imshow('frame',frame)
	cv2.imshow('gray',gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	rate.sleep()