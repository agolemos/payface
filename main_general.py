import cv2
import os
import dlib
import numpy as np
from modularized_code.model_ops import Ops
from preprocessing.data_preprocessing import Preprocessing
from data.data_acquisition import Acquisition
ops=Ops()


pp_obj=Preprocessing()
root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model_path = './shape_predictor_68_face_landmarks.dat'

# Load Anti-Spoofing Model graph
model =ops.get_model("lbp_model.m")
# load antispoofing model weights

print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

video = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
while True:
    #try:
        ret,frame = video.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image,1.3,5)
        for (x,y,w,h) in faces:

            face = frame[y-5:y+h+5,x-5:x+w+5]

            print('Teste')
            feature = pp_obj.lbp_features_online(frame)
            print(feature.shape)
            #preds = model.predict(feature.reshape(1, -1))
            preds = model.predict_proba(feature.reshape(1, -1))[0][1]
            print(preds)


            print(preds)
            if preds<0.5:
                label = 'Nao autorizado'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)
            else:
                label = 'Autorizado'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    #except Exception as e:
        #pass

video.release()
cv2.destroyAllWindows()