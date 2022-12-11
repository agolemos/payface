
import numpy as np
from modularized_code.model_ops import Ops
import os
from sklearn.model_selection import train_test_split
from metric.metric import Metric
from sklearn.model_selection import StratifiedKFold
import pandas as pd



class Modeling:
    train=''
    test=''
    m=''


    def __init__(self):
        self.m=Metric()
        self.ops=Ops()
        return


    def split_train_test(self, df, value ):

        self.train, self.test = train_test_split(df, test_size=value)




        return

    def StratifiedKFold(self, df, classifier):
        print("##### K-Fold test ######")
        cv = StratifiedKFold(n_splits=5)
        model = classifier
        X=np.array(df.iloc[:,:-1])
        y=np.array(df.iloc[:,-1:])
        cont=0
        for train_index, test_index in cv.split(X, y):
            cont=cont+1
            print("FOLDER {}".format(cont))
            train_feature, test_feature = X[train_index], X[test_index]
            train_label, test_label = y[train_index], y[test_index]
            model.fit(train_feature, train_label)
            predict_proba = model.predict_proba(train_feature)
            acc, eer, hter = self.m.general_metric(predict_proba, train_label)  # DESACOPLAR
            predict_proba = model.predict_proba(test_feature)
            acc, eer, hter = self.m.general_metric(predict_proba, test_label)
        return

    def lbp_train(self,classifier):

        print('******* Metricas de Treino ***************')

        train_feature=self.train.iloc[:,:-1]
        train_label = self.train.iloc[:,-1:]

        #print(train_feature.isnull().sum())

        print('Amount of data used for training: {}'.format(len(train_feature)))
        model = classifier
        model.fit(train_feature, train_label)

        data_dir = os.path.abspath(os.path.dirname(__file__))

        self.ops.save_model(model,"lbp_model.m")

        predict_proba = model.predict_proba(train_feature)
        predict = model.predict(train_feature)



        #resultado=train_label.copy()
        cont=0
        for i in range(0,len(predict)):

            if(predict[i]==float(train_label.iloc[i])):
                cont+=1
                print(predict[i],float(train_label.iloc[i]), predict_proba[i][1])
        print(cont, len(train_label))

        acc, eer, hter = self.m.general_metric(predict_proba, train_label) # DESACOPLAR


    def lbp_test_online(self,feature):
        return

    def lbp_test(self):
        print('******* Metricas de Teste ***************')
        test_feature = self.test.iloc[:, :-1]
        test_label = self.test.iloc[:, -1:]
        print('Amount of data used for testing: {}'.format(len(test_feature)))


        model = self.ops.get_model("lbp_model.m")
        predict_proba = model.predict_proba(test_feature)
        predict = model.predict(test_feature)
        acc, eer, hter = self.m.general_metric(predict_proba, test_label)







