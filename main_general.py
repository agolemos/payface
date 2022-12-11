

import cv2
from modularized_code.model_ops import Ops
from preprocessing.data_preprocessing import Preprocessing
from modeling.resnet import ResNet50
from utils.utils import Utils



ops=Ops()
pp_obj=Preprocessing()
utils=Utils()




model =ops.get_model("lbp_model.m") # Modelo LBP
resnet = ResNet50(ops.get_model_path('013__restnet50__256_256__grandtest_minus002videos')) #Modelo RESNET




def is_attack_lbp():
    feature = pp_obj.lbp_features_online(frame)
    preds = model.predict_proba(feature.reshape(1, -1))[0][1]
    if preds < 0.5:
        return True
    else:
        return False

def is_attack_resnet():
    is_attack = resnet.is_attack(frame)
    if is_attack:
        return True
    else: return False



# For webcam input:
cap = cv2.VideoCapture(0)


while cap.isOpened():

    success, frame = cap.read()
    if not success: continue
    image_rows, image_cols, _ = frame.shape

    try:
        rect_start_point,rect_end_point = utils.MediaPipeFaceDetection(frame)
        x, y = rect_start_point
        w, h = rect_end_point
        is_attack = is_attack_resnet()

        if is_attack:
            label = 'Nao autorizado'
            cv2.putText(frame, label[::-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 0, 255), 2)
        else:
            label = 'Autorizado'
            cv2.putText(frame, label[::-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 255, 0), 2)

        cv2.imshow('Liveness Detection', cv2.flip(frame, 1))
        key = cv2.waitKey(1)
        if key == ord('q'): break

    except:
        cv2.imshow('Liveness Detection', cv2.flip(frame, 1))
        key = cv2.waitKey(1)
        if key == ord('q'): break

cap.release()