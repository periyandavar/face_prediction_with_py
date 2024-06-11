import cv2
import os
import numpy as np
from Tkinter import Tk
from tkFileDialog import askopenfilename

sub=['']
idl=1

def detect_face(img):
  """Fuction detect and return the face part of the passed image"""
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face_casecade=cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
  faces=face_casecade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5)
  if len(faces)==0:
   return None,None
  (x,y,w,h)=faces[0]
  return gray[y:y+w,x:x+h],faces[0]
  
def train(path):
  global idl,sub
  """Preparing the Dataset for the identification of the person """
  print("Loading,...! Prepering Dataset,...! ")
  dirs=os.listdir(path)
  labels=[]
  faces=[]
  for dire in dirs:
   if not dire.startswith('s'):
     continue
   label=idl
   spl=dire.split(" ",1)
   print(dire)
   #sub[idl]=dire[1]
   sub.append(spl[1])
   print(idl)
   print(dire)
   idl=idl+1
   #label=int(dire.replace('s',''))
   sub_dir_path=path+'/'+dire
   sub_dirs=os.listdir(sub_dir_path)
   for sub_dir in sub_dirs:
    if sub_dir.startswith('.'):
     continue
    img_path=sub_dir_path+'/'+sub_dir
    image=cv2.imread(img_path)
    print(" ........ "+img_path)
    face,ret=detect_face(image)
    if face is not None:
      labels.append(label)
      faces.append(face)

      
  return faces,labels
   
 
def draw_rect(img,rect):
 '''Fuction to draw a rectangle arround the founded face part of the image'''
 (x,y,w,h)=rect
 cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),2)


def draw_text(img,text,x,y):
 '''print the person name to the image '''
 cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)


def predict(test_img):
 '''Function to predict the person'''
 img=test_img.copy()
 face,rect=detect_face(img)
 label,conf=face_recognizer.predict(face)
 print(sub)
 print(label)
 name=sub[label]
 draw_rect(img,rect)
 draw_text(img,"hi "+name+" !",rect[0],rect[1]-5)
 return img,conf


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Identifying the Person ")
print("Preparing the Datasets,...")
faces,labels=train("Database")
face_recognizer.train(faces,np.array(labels))
print("Dataset is created successfully,..!")
print("System Trained with "+str(len(labels))+" Persons and "+str(len(faces)) + "images ,..")
print("please choose an image ...")
Tk().withdraw()
filename=askopenfilename()
print("Predicting the persons ,...!")

test1=cv2.imread(filename)
#test1=cv2.imread("test-data/test.jpg")
predicted,conf=predict(test1)
print("prediction completed Successfully")
#cv2.imshow("output",predicted)
print(conf)
if(conf<65):
 cv2.imshow("output ", cv2.resize(predicted, (400, 500)))
 cv2.waitKey(0)
 cv2.destroyAllWindows()
else:
 print("unable to predict the person")
