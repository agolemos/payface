import mediapipe as mp
import cv2
import os
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_face_detection = mp.solutions.face_detection
mp_FaceKeyPoint = mp_face_detection.FaceKeyPoint
mp_get_key_point = mp_face_detection.get_key_point
mp_drawing = mp.solutions.drawing_utils
mp_normalized_to_pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates



class AWS_Utils:
    s3=''
    bucket=''
    cont=0
    region_name=''
    
    def __init__(self,bucket='payface-datasets', region_name='us-east-1' ):
        self.s3 = boto3.resource('s3', region_name)
        self.bucket = self.s3.Bucket(bucket)
        self.region_name=region_name
        return
    
    
    def get_names_s3(self, name_folder, bucket='payface-datasets', region_name='us-east-1'):
        
        folder = name_folder
        files_in_s3 = [f.key.split(folder + "/")[1] for f in self.bucket.objects.filter(Prefix=folder).all()]
        return files_in_s3
    
    def read_image_from_s3(self, key, bucket='payface-datasets', region_name='us-east-1'):
   
        object = self.bucket.Object(key)
        response = object.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        return np.array(im)
    
    def write_image_to_s3(self,img_array, key, bucket='payface-datasets', region_name='us-east-1'):
       
        object = self.bucket.Object(key)
        file_stream = BytesIO()
        im = Image.fromarray(img_array)
        im.save(file_stream, format='jpeg')
        object.put(Body=file_stream.getvalue())
    
    

class Utils:
    
    dataset=''

    def __init__(self):

        return
    
    
    


    

    def MediaPipeFaceDetection(self,frame):
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=1) as face_detection:
            image_rows, image_cols, _ = frame.shape
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

            return rect_start_point, rect_end_point



    def detect_faces(self,img):

        img_height, img_width = img.shape[:2]
        faces = list()
        with mp_face_detection.FaceDetection(
                # model_selection=0,    #NOTE: this attribute doesn't exist, for some reason
                min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(img)
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                rel_x = relative_bounding_box.xmin
                rel_y = relative_bounding_box.ymin
                x, y = mp_normalized_to_pixel_coordinates(rel_x, rel_y, img_width, img_height)
                rel_w = relative_bounding_box.width
                rel_h = relative_bounding_box.height
                w, h = mp_normalized_to_pixel_coordinates(rel_w, rel_h, img_width, img_height)
                faces.append((x, y, w, h))
        return faces

    def select_largest_central_face(self,faces, img_height, img_width):
        def size_of_bounding_box(w, h):
            return w * h

        def distance_to_center(coord, length):
            normalized = coord / length
            return 1 - 2 * abs(normalized - 0.5)  # 1: center, 0: corner

        def sorting_logic(key):
            x, y, w, h = key
            size = size_of_bounding_box(w, h)
            d_x = distance_to_center(x, img_width)
            d_y = distance_to_center(y, img_height)
            return size, -(d_x * d_y)

        # Return the largest central face detected
        faces = sorted(faces, key=sorting_logic, reverse=True)
        x, y, w, h = faces[0]
        return x, y, w, h

    def expand_bounding_box(self, detection, resolution):
        x, y, w, h = detection
        min_width, min_height = resolution
        if h < min_height:
            y = max(y - (min_height - h) // 2, 0)
            h = min_height
        if w < min_width:
            x = max(x - (min_width - w) // 2, 0)
            w = min_width
        return x, y, w, h

    # FACE DETECTION: HIGH-ORDER LOGIC ########################################################
    def detect_face(self,img):
        img_height, img_width = img.shape[:2]
        # use a pre-trained model to detect faces
        faces = self.detect_faces(img)
        # handle case that more than one face is detected
        x, y, w, h = self.select_largest_central_face(faces, img_height, img_width)
        return x, y, w, h

    def crop_face(self,img, detection, resolution=None):
        x, y, w, h = detection
        # if resolution is non-null, adjust the bounding box
        if resolution is not None:
            assert all(_min <= _avail for _min, _avail in zip(resolution, img.shape))
            x, y, w, h = self.expand_bounding_box(detection, resolution)
        # crop
        return img[y:y + h, x:x + w]