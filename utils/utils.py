import sys
import cv2
import os

class Utils:
    
    dataset=''

    def __init__(self, ):

        return



    def convert_videos_to_images(self, name):

        # Importing all necessary libraries

        # Read the video from specified path
        cam = cv2.VideoCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")

        try:

            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')

        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')

        # frame
        currentframe = 0

        while (True):

            # reading from frame
            ret, frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = './data/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        return
