#!/usr/bin/env python3
import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No camera!")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False

print("r=Record, q=Quit, s=Shot")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    status = "REC" if recording else "STOP"
    cv2.rectangle(frame, (10,10), (150,40), (0,255,0) if recording else (0,0,255), -1)
    cv2.putText(frame, status, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow('Face Live', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        recording = not recording
        if recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(f"face_{ts}.mp4", fourcc, 30.0, (640,480))
            print("RECORDING...")
        else:
            out.release()
            out = None
            print("STOPPED")
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"shot_{ts}.jpg", frame)
        print("Screenshot OK")
    
    if recording and out: out.write(frame)

cap.release()
cv2.destroyAllWindows()
