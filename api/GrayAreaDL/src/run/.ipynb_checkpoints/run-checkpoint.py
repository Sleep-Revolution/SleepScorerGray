import os
import logging
import sys
sys.path.insert(0,os.getcwd())
# sys.path.insert(0, os.path.join(os.getcwd(),"./SleepScorerGray/api/GrayAreaDL"))

from datetime import datetime
from pathlib import Path
import copy

import numpy as np
import yaml

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline


from src.data.predictors import *
from src.data.prediction import *
from src.data.datagenerator import *

from src.preprocessing.harmonisation import *

from src.model.callbacks import *
from src.model.model import *
from src.model.mixturemodels import *

from src.utils.yamlutils import *
from src.utils.save_xp import *


from tensorflow import keras
import tensorflow_addons
import pandas as pd
import gc
############################ CONF POSSIBLE ############################

# python .\src\run\runAEWea.py .\params\01_alberta_VQVAESatWea.yaml

class RunTrain:
    """
    Run a training for the Beps-ia predictor based on the config stored at the path ``yaml_config_path``

    :param yaml_config_path: configuration file path
    :param append_date_to_logdir: create the run in a subfolder with the date
    :return: SaveXp instance
        Contain the metrics and predictions of the validation data at the end of training for each fold of the
        cross validation.

    """
    def __init__(self,yaml_config_path, append_date_to_logdir=True):
        # Parse yaml
        loader = get_conf_loader()
        with open(yaml_config_path) as file:
            params = yaml.load(file, Loader=loader)

        # Get instances from configuration file and build Pipelines
        logging.info("Config File is loaded")

        self.predictors: X = params["Data"]["Predictors"]
        self.prediction: Y = params["Data"]["Prediction"]

        self.pipeline_prepro = Pipeline([(str(estimator), estimator) for estimator in params["Preprocessing"]])

        signals_parts = self.predictors.getallpart()

        self.logpath = Path(params["log_dir"])
        self.yaml_config_path=yaml_config_path
        self.append_date_to_logdir=append_date_to_logdir
        self.params = params
        
        self.HM = Harmonisation(self.prediction,
                                self.predictors,
                                params["Harmonisation"]["pathTosave"],
                                windowEpoch=params["Harmonisation"]["windowEpoch"],
                                pipeline_prepro=self.pipeline_prepro )
        
        #Test list of data
        self.HM.ListLoaTraSav(signals_parts)
        # self.HM.ListLoaTraSav([1,7,10,25,49,50])
        # self.HM.ListLoaTraSav([1,7])
        
        self.List_ID = self.HM.List_ID
        self.signalsNames = self.HM.signalsNames
        

    def run_model(self):
        params = self.params
        logpath = self.logpath

        if self.append_date_to_logdir:
            #logpath /= f"{datetime.now():%Y%m%d-%H%M%S}_run"
            logpath /= params["nameDir"]

        xp_saver_ae = SaveXp(logpath)
        xp_saver_ae.save_experiment_metadata(self.yaml_config_path)

        
        cv = ShuffleSplit(n_splits=params["Splitters"]["n_splits"],
                              test_size=params["Splitters"]["test_size_percent"]
                             )
        
        for train_indexes, valid_indexes in cv.split(X=self.List_ID):
            training_generator = DataGenerator(train_indexes,
                                     params["Harmonisation"]["pathTosave"],
                                     params["DataGenerator"]["batch_size"],
                                     self.signalsNames, 
                                     self.prediction.predictionName, 
                                     params["DataGenerator"]["Numscorer"],
                                     shuffle=False,
                                     random_state=17)
            validation_generator = DataGenerator(valid_indexes,
                         params["Harmonisation"]["pathTosave"],
                         params["DataGenerator"]["batch_size"],
                         self.signalsNames, 
                         self.prediction.predictionName, 
                         params["DataGenerator"]["Numscorer"],
                         shuffle=False,
                         random_state=17)



            path_weights = Path(logpath) / "_best_model_AE.hdf5"
            # Perfcallback = PerformancePlotCallback(validation_generator, "VQVAE", logpath, 0)

            cb_list = generate_callbacks(logpath, params["Callbacks"]["patience"],
                                         path_weights,
                                         metric_monitored=params["Callbacks"]["callbacks_metrics"])
            # cb_list.append(Perfcallback)

            modelae: modelae_builder = params["Model"]
            modelae.compile()

            print(modelae.summary())

            modelae.fit(training_generator,
                      epochs=params["Callbacks"]['epochs'],
                      validation_data=validation_generator,
                      callbacks=cb_list,
                      verbose=1
                      )
            modelae.evaluate(validation_generator)

            if self.append_date_to_logdir:
                logpath /= f"{datetime.now():%Y%m%d-%H%M%S}"
            xp_saver = SaveXp(logpath)
            xp_saver.save_experiment_metadata(self.yaml_config_path)

        # del train_set
        del valid_set
        del training_generator
        del validation_generator
        del train_set
        xp_saver.save_xp()
        return xp_saver


class RunPredict:
    def __init__(self, CONFPATH, ):
        self.SCORE_DICT = {
            'Wake': 0.,
            'N1': 1.,
            'N2': 2.,
            'N3': 3.,
            'REM': 4.}
            
        loader = get_conf_loader()
        with open(CONFPATH) as file:
            paramsPred = yaml.load(file, Loader=loader)
            
        files = os.listdir(paramsPred["ModelPath"])
        confyml = [i for i in files if i.split(".")[-1]=="yaml"][0]
        logging.basicConfig(level=logging.INFO)
        loader = get_conf_loader()
        with open(os.path.join(paramsPred["ModelPath"],confyml)) as file:
            params = yaml.load(file, Loader=loader)
        self.pipeline_prepro = Pipeline([(str(estimator), estimator) for estimator in params["Preprocessing"]])
        
        with open(CONFPATH) as f:
            config = yaml.load(f, Loader=loader)
        self.model = keras.models.load_model(paramsPred["ModelPath"])
        print(self.model.summary())
        logging.info("Model Loaded")
        
        self.generator = DataGeneratorPred(paramsPred["EDFPath"],params["Data"]["Predictors"].signalsNames,pipeline=self.pipeline_prepro)
        self.nb_samples = len(self.generator.list_id)
        self.paramsPred = paramsPred
        
        
    def predict(self):
        logging.info("Start Prediction")
        # steps = self.nb_samples
        return self.model.predict(self.generator,steps =  1)

def GenerateMultiSamp(x,E):
    x = np.array([x]).astype(np.float64)
    if sum(x.sum(axis=1)) != 1:
        x[0,:] = x[0,:]/sum(x[0,:])

    gen = GenMixtSampleFromCatEns(E,x)
    X,Z = gen.generate(2,distribution="Multinomial")
    return X[0,:].tolist()
    
    
def main():
    # logging.basicConfig(level=logging.DEBUG)
    yaml_path = str(sys.argv[1])
    
    if len(sys.argv)>2:
        if str(sys.argv[2])!="pred":
            run_pipeline = RunTrain(yaml_path)
            run_pipeline.run_model()
        else:
            run_pipeline = RunPredict(yaml_path)
            y = run_pipeline.predict()
            listFile = os.listdir(run_pipeline.paramsPred["PredPath"])
            if len(listFile)==0:
                print("Empty Dir") 
            else:
                for f in listFile:
                    try:
                        os.remove(os.path.join(run_pipeline.paramsPred["PredPath"],f))
                    except Exception as e: 
                        print(f"WARNING: {e}")
                        
                print("Dir cleared successfully")
            for i in range(y.shape[0]):
                filepath = os.path.join(run_pipeline.paramsPred["PredPath"],run_pipeline.generator.Predictors_.allEdf[i]+".csv")
                
                Y_MM = np.fromiter(map(lambda x : GenerateMultiSamp(x,E=1000),y[i,:,:]), dtype=np.dtype((int, len(run_pipeline.SCORE_DICT))))
                MMM = MixtModel(E=1000,distribution="Multinomial",filtered=True,threshold=0.3)
                MMM.fit(Y_MM)
                Z_G = MMM.clusters
                Z_G = (Z_G != (-1))*1
                warnings = {"10":[],"30":[],"60":[],"120":[]}
                results = np.concatenate((y[i,:,:],Z_G[np.newaxis].T),axis=1)
                for k in list(warnings.keys()):
                    Nrow = int(Z_G.shape[0]/(int(k)*2))
                    Ncol = int(int(k)*2)
                    tmp = Z_G.reshape((Nrow,Ncol)).sum(axis=1)
                    warnings[k] = np.tile(tmp,(Ncol,1)).T.reshape(Ncol*Nrow)
                    results = np.concatenate((results,warnings[k][np.newaxis].T),axis=1)
                
                pd.DataFrame(results,columns = [run_pipeline.SCORE_DICT.keys(),"GrayArea"]+["Warning "+k for k in list(warnings.keys())]).to_csv(filepath)
            
    else:
        run_pipeline = RunTrain(yaml_path)
        run_pipeline.run_model()



if __name__ == "__main__":
    main()
