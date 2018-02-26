import numpy as np
import cv2
import serial

cap = cv2.VideoCapture(0)                        # capture from web cam
fgbg = cv2.createBackgroundSubtractorMOG2()      # background subtraction
ser = serial.Serial()                            # initialise serial communication   

ser.port = 'COM6'                                # connect to port 
ser.baudrate = 9600                              # set baud rate  
print(ser)
ser.open()                                       # start serial communication

preval,z,curval=0,0,0

while(1):
    ret, frame = cap.read()                        

    fgbg.setDetectShadows(False)                            # ignore shadows
    foreground = fgbg.apply(frame,learningRate = 0.0001)    # extract foreground

    kernel = np.ones((3, 3), np.uint8)                      # set kernel size for erosion and dilation  
    erode = cv2.erode(foreground, kernel, iterations=1)
    dilate = cv2.dilate(erode,kernel,iterations = 1)        # reduce 
    blur = cv2.medianBlur(erode,5)                          # the       
    erode = cv2.erode(blur, kernel, iterations=1)           # noise
    blur = cv2.medianBlur(erode,5)
    
    ret , th = cv2.threshold(blur,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                  # set threshold for better contour extraction 
    ran , contours , hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   # find contours 
    
    if len(contours)==0:                              # discontinue if no contours
        continue
    c = max(contours , key = cv2.contourArea)         # finds the largest contour
    im = cv2.drawContours(frame,[c],-1,(0,255,0),3)   # draws contour

    hull = cv2.convexHull(c,returnPoints=False)       # finds hull coordinates 
    defects = cv2.convexityDefects(c,hull)            # finds defects coordinates 
    print (defects)
    if defects is None:                               # discontinue if defects is none
        continue

    sum2=0
    sum1=0
    preval = z
    z=0

    for i in range(defects.shape[0]):                # ennumerate the defect points   
        s,e,f,d = defects[i,0]                       
        
        if d>3000:                                   # check the distance of far point
            start = tuple(c[s][0])                    
            end = tuple(c[e][0])
            far = tuple(c[f][0])
            cv2.line(im,start,end,[255,0,0],2)       # draw line from start to end points
            cv2.circle(im,far,5,[0,0,255],-1)        # draw circle on far points  
            cv2.circle(im,start,5,[0,255,255],-1)    # draw circle on start points 
            sum1=sum1+c[f][0][0]
            sum2=sum2+c[f][0][1]
            z+=1
    curval = z                                       # number of fingers   
    if preval == curval:
        pass
    else:
        if z<7 :
            print('no of fingers:',z-1)
    
    if (z-1)==1:       
        ser.write(b'1')
    elif (z-1)==2:                #
        ser.write(b'2')           #
    elif (z-1)==3:                #  send data through serial port   
        ser.write(b'3')           #
    elif (z-1)==4:                #
        ser.write(b'4')
    elif (z-1)==5:    
        ser.write(b'5')

    cv2.putText(frame,str(z-1),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)                                   
    cv2.imshow('hand detection',blur)
    cv2.imshow('background subtraction',frame)
    cv2.imshow('background subtraction1',foreground)

    key = cv2.waitKey(1)
 
    if key == ord("q"):
        break

ser.close()
cv2.destroyAllWindows()
