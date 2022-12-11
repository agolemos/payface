

from data.data_acquisition import Acquisition
from preprocessing.data_preprocessing import Preprocessing
from modeling.modeling import Modeling
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(os.getcwd())
print(os.getcwd())

if __name__ == '__main__':


# 1º ETAPA: AQUISIÇÃO DO DATASET

    msu_obj=Acquisition('MSU')
    #msu_obj.get_videos_img()

# 2º ETAPA: PRE-PROCESSAMENTO DO DATASET

    pp_obj=Preprocessing()

    #df_featured=pp_obj.lbp_features(msu_obj)

    df_featured=pp_obj.get_lbp_features(msu_obj)
    #print(df_featured.iloc[0,:-1].shape)

# 3º ETAPA: DIVISÃO TREINO E TESTE
    model_obj=Modeling()
    model_obj.split_train_test(df_featured,0.33)

# 4º ETAPA: SELEÇÃO DO MODELO DE CLASSIFICAÇÃO

    #classifier = SVC(kernel='rbf', C=1e3, gamma=0.5, class_weight='balanced', probability=True)
    classifier = MLPClassifier(random_state=1, max_iter=5000)
    classifier_name="log_mlp"

# 5º ETAPA: TREINO
    mlflow.set_experiment("LBP_EXPERIMENT")
    with mlflow.start_run():
        model_obj.lbp_train(classifier)
        mlflow.sklearn.log_model(classifier,classifier_name)
        print("Model run: ",mlflow.active_run().info.run_uuid)
    mlflow.end_run()

# 6º ETAPA: TESTE
    model_obj.lbp_test()

#model_obj.StratifiedKFold(df_featured, classifier)










