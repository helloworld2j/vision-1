import cv2
from deepface import DeepFace
import threading

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter=0

face_m= False

ref=cv2.imread("E:\opencv\WIN_20240127_18_23_21_Pro.jpg")

def face_check(frame):
    global face_m
    try:
            if(DeepFace.verify(frame,ref.copy()))["verified"]:
                face_m=True
            else:
                face_m=False
    except ValueError:
        face_m=False
    
while True:
    ret,frame=cap.read()

    if ret:
        if counter % 30 == 0 :
            try:
                threading.Thread(target=face_check,args=(frame.copy,))
            except ValueError:
                pass
        counter+=1

        if face_m:
            cv2.putText(frame,"MATCH",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,250,0),3)
        else:
            cv2.putText(frame,"NO MATCH FOUND",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,250,0),3)
        
        cv2.imshow("video",frame)

    key=cv2.waitKey(1)
    if key== ord("q"):
        break
cv2.destroyAllWindows()



