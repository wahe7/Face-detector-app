import cv2
# for random color of rectangle
from random import randrange


# load some pre-trained data on face frontal from opencv
# harr is inventor  and cascade is chain of event
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# choose image to detect face in
# only one person
# img=cv2.imread('dhoni.jpg')

# 2nd image
# more than one person
#img=cv2.imread('pair.jfif')

# to capture video from webcam
# 1 is for specific webcam 0 is for default
webcam=cv2.VideoCapture(1)

#iterate forever over frames
while True:

     #read the current frame
     succsessful_frame_read,frame=webcam.read()

     #Must convert to grayscale 
     grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

     # detect faces
     face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

     
     #Draw rectangle around the faces
     #use of loop
     for(x,y,w,h) in face_coordinates:
          cv2.rectangle(frame , (x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

     cv2.imshow('wahe face detector', frame)
     # here 1 is waut for milisecond
     key = cv2.waitKey(1)

     # stop if Q key is pressed

     if key==81 or key==113: #ascii value of Q and q.
          break

     # release webcam
webcam.relaese()


"""  





# To print coordinates

# print(face_coordinates)

# to show image
cv2.imshow('wahe face detector', img)
# until we press any key it will not close
cv2.waitKey()


print("code complete")
"""
