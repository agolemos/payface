
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import cv2
from modularized_code.model_ops import Ops
from preprocessing.data_preprocessing import Preprocessing


ops=Ops()
pp_obj=Preprocessing()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
model =ops.get_model("lbp_model.m")

# For webcam input:
cont=0
cap = cv2.VideoCapture(0)
tag=0

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=1) as face_detection:
  while cap.isOpened():

    try:
        success, frame = cap.read()

        image_rows, image_cols, _ = frame.shape
        if not success:

          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        if results.detections:


            detection = results.detections[0]
            location = detection.location_data
            relative_bounding_box = location.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin,
                                                                image_cols, image_rows)
            rect_end_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width,
                                                              relative_bounding_box.ymin + relative_bounding_box.height,
                                                              image_cols, image_rows)
            x, y = rect_start_point
            w, h = rect_end_point


            feature = pp_obj.lbp_features_online(frame)
            preds = model.predict_proba(feature.reshape(1, -1))[0][1]
            print(preds)

            if preds < 0.5:
                label = 'Nao autorizado'
                tag=0
                cv2.putText(frame, label[::-1], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, rect_start_point, rect_end_point,
                              (0, 0, 255), 2)
            else:
                label = 'Autorizado'
                tag=1
                cv2.putText(frame, label[::-1], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, rect_start_point, rect_end_point,
                              (0, 255, 0), 2)


            cv2.imshow('MediaPipe Face Detection', cv2.flip(frame, 1))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:

            cv2.imshow('MediaPipe Face Detection', cv2.flip(frame, 1))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


    except:

        continue

cap.release()