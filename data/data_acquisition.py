#import sys
#sys.path.insert(0, '/home/ec2-user/environment/payface')
import matplotlib.pyplot as plt
#import cv2
import glob

import os
import pandas as pd
import splitfolders
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
from utils.utils import AWS_Utils 
import io


class Acquisition:
    
    data_dir = os.path.abspath(os.path.dirname(__file__))
    user_dir = '/home/ec2-user/environment/payface/user/Thiago Lemos'
    name_dataset=''
    raw_dir = inter_dir = featured_dir = path=''
    train_dir = test_dir = val_dir = ''
    p_train=p_test=p_val=''
    df=''
    AWS=''
    utils=''




    def __init__(self, name_dataset):
        self.utils=AWS_Utils()
        self.AWS=True
        #=utils.get_names_s3('MSU-MFSD/unprocessed/attack-imgs')
        #r=utils.read_image_from_s3('MSU-MFSD/labeled-imgs/ipad/attack_client001_android_SD_ipad_video_scene01-00015.png')
       

        self.name_dataset=name_dataset
        if self.name_dataset == 'MSU-MFSD':
            if self.AWS==False:
                print('Entrou 1')
                self.data_dir = os.path.join(self.data_dir, 'dataset/dataset-msu-imgs/')
            else:
                print('Entrou 2')
                self.data_dir = os.path.join(self.name_dataset, 'unprocessed')
                
        self.path = os.path.join(self.user_dir, 'data_user')
        self.raw_dir = os.path.join(self.path, self.name_dataset, 'Raw')
        self.inter_dir = os.path.join(self.path, self.name_dataset, 'Intermediate')
        self.featured_dir = os.path.join(self.path, self.name_dataset, 'Featured')
        print(self.data_dir)
        #self.data_file_delete()
        self.data_file_creator()
        self.PathDataFrameAWS()

        return
    
    
    def PathDataFrameAWS(self):
        
        
        classes_dir=['attack-imgs','real-imgs']
        val_ratio = 0.20
        test_ratio = 0.20
        contador=0
        df=pd.DataFrame()
        df = pd.DataFrame({'path': pd.Series(dtype='str'),
                   'target': pd.Series(dtype='float')})
        for cls in classes_dir:
            #print(cls,contador)
            # Creating partitions of the data after shuffeling
            #print("$$$$$$$ Class Name " + cls + " $$$$$$$")
            src = self.data_dir + "/" + cls  # Folder to copy images from
            #print(src)
            files=[]
            allFileNames = self.utils.get_names_s3(src)
            #print(allFileNames)
            for i in range(0,len(allFileNames)):
                files.append(os.path.join(src, allFileNames[i]))
            np.random.shuffle(files)
            aux=pd.DataFrame(files)
            aux['target']=contador
            if(contador==0):
                df=aux.copy()
            else:
                df=df.append(aux)

            contador+=1
            
        #df.columns=['path','target']
        #df = df.sample(frac=1).reset_index(drop=True)
        
        df.columns=['path','target']
        self.df=df.copy()
        df.to_csv(os.path.join(self.raw_dir, "path_file.csv"), index=False)
        
        
        return 



# ***************************************************************************************************************
    def PathDataFrame (self):
        #print("########### Train Test Val Script started ###########")
        # data_csv = pd.read_csv("DataSet_Final.csv") ##Use if you have classes saved in any .csv file
        classes_dir=[]
        root_dir = self.inter_dir
        print("*****************************")
        print(os.listdir(self.data_dir))
        print("*****************************")
        for category in os.listdir(self.data_dir): classes_dir.append(category)


        processed_dir = self.data_dir

        val_ratio = 0.20
        test_ratio = 0.20
        contador=0
        df=pd.DataFrame()

        for cls in classes_dir:
            print(cls,contador)
            # Creating partitions of the data after shuffeling
            #print("$$$$$$$ Class Name " + cls + " $$$$$$$")
            src = processed_dir + "//" + cls  # Folder to copy images from
            #print(src)
            files=[]
            allFileNames = os.listdir(src)
            for i in range(0,len(allFileNames)):
                files.append(os.path.join(src, allFileNames[i]))
            np.random.shuffle(files)
            aux=pd.DataFrame(files)
            aux['target']=contador
            if(contador==0):
                df=aux.copy()
            else:
                df=df.append(aux)

            contador+=1
        df.columns=['path','target']
        #df = df.sample(frac=1).reset_index(drop=True)
        self.df=df.copy()
        df.to_csv(os.path.join(self.raw_dir, "path_file.csv"), index=False)

        return

# ***************************************************************************************************************
    def getPathDataframe(self):
        return self.df
# ***************************************************************************************************************
    def data_file_creator(self):
        try:

            os.mkdir(os.path.join(self.path, self.name_dataset))
            os.mkdir(self.raw_dir)
            os.mkdir(self.inter_dir)
            os.mkdir(self.featured_dir)

        except:
            flag=0

        return

# ***************************************************************************************************************

    def data_file_delete(self):
        try:
            shutil.rmtree(os.path.join(self.path, self.name_dataset))
        except:
            flag=0
        return




    def getTrainPath(self, pasta):


        if(os.path.exists(self.raw_dir)):
            if pasta =='Raw':
                imdir=os.path.join(self.raw_dir,'train')
            if pasta == 'Intermediate':
                imdir=os.path.join(self.inter_dir,'train')
            if pasta =="Featured":
                imdir=os.path.join(self.featured_dir,'train')

            classes = os.listdir(imdir)
            cont=-1
            general=pd.DataFrame()

            for c in classes:
                lista = []
                cont=cont+1
                path=os.path.join(imdir,c)
                img= os.listdir(path)
                for x in img: lista.append(os.path.join(path,x))
                df=pd.DataFrame(lista)
                df['target']=cont
                general=general.append(df)


            general.columns=['path','target']
            return general
        else:
            return -1

# ***************************************************************************************************************
    def getTestPath(self, pasta):


        if(os.path.exists(self.raw_dir)):
            if pasta =='Raw':
                imdir=os.path.join(self.raw_dir,'test')
            if pasta == 'Intermediate':
                imdir=os.path.join(self.inter_dir,'test')
            if pasta =="Featured":
                imdir=os.path.join(self.featured_dir,'test')

            classes = os.listdir(imdir)
            cont=-1
            general=pd.DataFrame()
            for c in classes:
                cont=cont+1
                path=os.path.join(imdir,c)
                img= os.listdir(path)
                df=pd.DataFrame(img)
                df['target']=cont
                general=general.append(df)


            general.columns=['path','target']
            return general
        else:
            return -1


# ***************************************************************************************************************
    def get_lbp_features(self):
        dir=os.path.join(self.inter_dir,"lbp_feature.npy")
        feature_label = np.load(dir)
        return feature_label[:, :-1], feature_label[:, -1].astype(np.uint8)









# ***************************************************************************************************************
    def train_test_split_original (self):
        print("########### Train Test Val Script started ###########")
        # data_csv = pd.read_csv("DataSet_Final.csv") ##Use if you have classes saved in any .csv file
        classes_dir=[]
        root_dir = self.inter_dir
        for category in os.listdir(self.data_dir): classes_dir.append(category)


        processed_dir = self.data_dir

        val_ratio = 0.20
        test_ratio = 0.20
        contador=0

        for cls in classes_dir:
            # Creating partitions of the data after shuffeling
            print("$$$$$$$ Class Name " + cls + " $$$$$$$")
            src = processed_dir + "//" + cls  # Folder to copy images from

            allFileNames = os.listdir(src)
            np.random.shuffle(allFileNames)
            train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                      [int(len(allFileNames) * (
                                                                                  1 - (val_ratio + test_ratio))),
                                                                       int(len(allFileNames) * (1 - val_ratio)),
                                                                       ])

            train_FileNames = [src + '//' + name for name in train_FileNames.tolist()]
            val_FileNames = [src + '//' + name for name in val_FileNames.tolist()]
            test_FileNames = [src + '//' + name for name in test_FileNames.tolist()]

            print('Total images: ' + str(len(allFileNames)))
            print('Training: ' + str(len(train_FileNames)))
            print('Validation: ' + str(len(val_FileNames)))
            print('Testing: ' + str(len(test_FileNames)))

            # # Creating Train / Val / Test folders (One time use)
            os.makedirs(root_dir + '/train//' + cls)
            os.makedirs(root_dir + '/val//' + cls)
            os.makedirs(root_dir + '/test//' + cls)

            # Copy-pasting images
            for name in train_FileNames:
                shutil.copy(name, root_dir + '/train//' + cls)

            for name in val_FileNames:
                shutil.copy(name, root_dir + '/val//' + cls)

            for name in test_FileNames:
                shutil.copy(name, root_dir + '/test//' + cls)

        print("########### Train Test Val Script Ended ###########")


# ***************************************************************************************************************



    def get_videos_img(self):
        pastas=[]
        A = []
        R = []
        for p in os.listdir(self.data_dir): pastas.append(os.path.join(self.data_dir, p))
        for i in range(len(pastas)):
            pasta = pastas[i]

            arquivos = os.listdir(pasta)

            for arquivo in arquivos:
                path = os.path.join(pasta, arquivo)
                if '.mp4' in path:

                    if 'attack' in pasta:
                        A.append(path)
                    elif 'real' in pasta:
                        R.append(path)





        return




