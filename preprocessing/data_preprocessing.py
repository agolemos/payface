import matplotlib.pyplot as plt
import cv2
import glob
import os
import pandas as pd
import splitfolders
from utils.utils import Utils
import numpy as np
from skimage import feature as skif
import cv2
import time
from utils.utils import Utils
#import pyprind
import sys
import time



class Preprocessing:


    def __init__(self):

        return


    def lbp_histogram(self, image, P=8, R=1, method='nri_uniform'):

        lbp = skif.local_binary_pattern(image, P, R, method)  # lbp.shape is equal image.shape
        # cv2.imwrite("lbp.png",lbp)
        max_bins = int(lbp.max() + 1)  # max_bins is related P
        hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
        return hist

    def get_lbp_features(self,dataset ):
        path = os.path.join(dataset.featured_dir, 'lbp_feature.npy')

        try:

            df = pd.DataFrame(np.load(path))

            return df


        except:
            print('There is no feature file!')
            exit(1)


    def lbp_features_online(self, face):
        image = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

        y_h = self.lbp_histogram(image[:, :, 0])  # y channel
        cb_h = self.lbp_histogram(image[:, :, 1])  # cb channel
        cr_h = self.lbp_histogram(image[:, :, 2])  # cr channel
        feature = np.concatenate((y_h, cb_h, cr_h))

        return feature

    def lbp_features(self, dataset):
        print('########## Extracting features (Loading) ###############')
        feature_label = []

        for index, line in dataset.df.iterrows():
            image_path = line['path']
            label = line['target']
            #print(index,label)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_h = self.lbp_histogram(image[:, :, 0])  # y channel
            cb_h = self.lbp_histogram(image[:, :, 1])  # cb channel
            cr_h = self.lbp_histogram(image[:, :, 2])  # cr channel
            feature = np.concatenate((y_h, cb_h, cr_h))
            feature_label.append(np.append(feature, np.array(label)))


        np.save(os.path.join(dataset.featured_dir,'lbp_feature.npy'), np.array(feature_label))
        print('########## Extracting features (Finished) ###############')

        return pd.DataFrame(feature_label)




