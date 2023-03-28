import os
import logging
import sys
sys.path.insert(0,os.getcwd())
# sys.path.insert(0, os.path.join(os.getcwd(),"./api/GrayAreaDL"))
# curl -X 'POST' 'http://130.208.209.67:80/nox-to-edf get_active_recording_time=false&get_all_scorings=false&export_scoring=true' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'nox_zip=@sas3nightTestSmall.zip;type=application/x-zip-compressed' -o zipped_edf.zip
# curl -X 'POST' 'http://130.208.209.67:80/nox-to-edf' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'nox_zip=@sas3nightTestSmall.zip;type=application/x-zip-compressed' -o zipped_edf.zip

from datetime import datetime
from pathlib import Path
import copy

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import numpy as np
import yaml

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

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
    def __init__(self, CONFPATH):
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
        self.paramsPred = paramsPred
        self.nsignals = len(self.paramsPred["SignalChannels"])
        self.all = self.paramsPred.get("ALL",False)
        self.ensemble = self.paramsPred.get("Ensemble",False)
        self.type_study = self.paramsPred.get("Type_study","PSG")
        
        if self.ensemble:
            self.generator = DataGeneratorPred(self.paramsPred["EDFPath"],
                                        self.paramsPred["SignalChannels"],
                                        pipeline=self.pipeline_prepro,
                                        ensemble = self.ensemble,
                                               type_study=self.type_study)
        else:
            self.generator = DataGeneratorPred(self.paramsPred["EDFPath"],
                                self.paramsPred["SignalChannels"],
                                pipeline=self.pipeline_prepro,
                                               type_study=self.type_study)
        self.nfile = len(self.generator.list_id)
    
    #################################     UNCOMMENT  ONLY FOR VALIDATION          ###########################################################
    # def MathiasValidation(self,file):
    #     i = file
    #     generator = MatiasGeneratorPred(self.paramsPred["EDFPath"])
    #     y = self.model.predict(generator.__getitem__(i),steps = 1)
    #     return y
    ###################################################################################################################

    def PredictToCSV(self,file):
        i = file
        y = self.model.predict(self.generator.__getitem__(i),steps = 1)
        Y = y.copy()
        # print("After prediction",y.shape)
        times = self.generator.currentSignal.metadata["TimeFromStart"]
        nepochs = y.shape[1]
        lenSignal = nepochs*int(30/self.generator.Predictors_.times_stamps)
        
        if lenSignal != times.shape[0]:
            times = times[:lenSignal]

        times = times.reshape((nepochs,int(30/self.generator.Predictors_.times_stamps)))
        times = times[:,0]

        # if isinstance(y,(np.ndarray)):
        #     y = y.tolist()

        if self.ensemble:
            Y = np.sum(Y, axis = 0)
            Y = Y/Y.sum(axis=1,keepdims=True)
            Hp_pred = np.argmax(Y, axis=1)

        Hp_pred = np.argmax(Y,axis=1)


        ####################### UNCOMMENT ONLY FOR VALIDATION ################################
#         y_valid = self.MathiasValidation(i)

#         gaborder = np.array(self.generator.currentSignal.metadata["SignalName"])
#         matorder = np.array(['AF3-E3E4','AF4-E3E4','AF7-E3E4','AF8-E3E4','E3-AFZ','E2-AFZ','E1-E4','E2-E3'])

#         all_ind = []
#         for h in range(self.nsignals):
#             k = np.where(gaborder[h]==matorder)[0][0]
#             Y_tmp = np.array(y[h])
#             Y_tmp = normalize(Y_tmp,norm="l1")
#             Hp_predtmp = np.argmax(Y_tmp,axis=1)
#             hg_final = np.argmax(y_valid[k,:,:], axis=1)
#             ind = np.where(Hp_predtmp != hg_final)[0]
#             all_ind.append(ind)
#             print(gaborder[h],matorder[k],ind.shape)
#         y_sum = np.sum(y_valid, axis = 0)
#         hg_final = np.argmax(np.sum(y_valid, axis = 0), axis=1)
#         ind = np.where(Hp_pred != hg_final)[0]
#         print("Validation, Number of divergence=",len(ind))
        ############################################################################################
        
        SignalName = np.array(self.generator.currentSignal.metadata["SignalName"])
        filepath = os.path.join(self.paramsPred["PredPath"],self.generator.Predictors_.allEdf[i-1]+".csv")

        Y_MM = np.fromiter(map(lambda x : self.GenerateMultiSamp(x,E=1000),Y), dtype=np.dtype((int, len(self.SCORE_DICT))))
        MMM = MixtModel(E=1000,distribution="Multinomial",filtered=True,threshold=float(self.paramsPred["GrayAreaThreshold"]))
        MMM.fit(Y_MM)
        Z_G = MMM.clusters
        Z_G = (Z_G != (-1))*1
        warnings = {"10":[],"30":[],"60":[],"120":[]}
        results = np.concatenate((Hp_pred[np.newaxis].T,Y,Z_G[np.newaxis].T),axis=1)
        for k in list(warnings.keys()):

            if Z_G.shape[0] % (int(k)*2) != 0:
                Nrow = int(Z_G.shape[0] / (int(k)*2))+1
                padd = Nrow*(int(k)*2)
                Z_G_tmp = np.zeros(padd)
                Z_G_tmp[:Z_G.shape[0]] = Z_G

                Nrow = int(Z_G_tmp.shape[0]/(int(k)*2))
                Ncol = int(int(k)*2)
                tmp = Z_G_tmp.reshape((Nrow,Ncol)).sum(axis=1)
                tmp = np.tile(tmp,(Ncol,1)).T.reshape(Ncol*Nrow)
                warnings[k] = tmp[:Z_G.shape[0]]
                results = np.concatenate((results,warnings[k][np.newaxis].T),axis=1)

            else:
                Nrow = int(Z_G.shape[0]/(int(k)*2))
                Ncol = int(int(k)*2)
                tmp = Z_G.reshape((Nrow,Ncol)).sum(axis=1)
                warnings[k] = np.tile(tmp,(Ncol,1)).T.reshape(Ncol*Nrow)
                results = np.concatenate((results,warnings[k][np.newaxis].T),axis=1)
        results = np.concatenate((times[np.newaxis].T,results),axis=1)
        print(f"Save: {filepath}")
        print("------------------------------------------------------------ END PREDICTION -------------------------------------------------------------")
        if ((self.all) & (self.ensemble)):
            columns = ["Times","Ens_Hypno"]+["Ens_"+k for k in list(self.SCORE_DICT.keys())]+["GrayArea"]+["Warning_"+k for k in list(warnings.keys())]
            DF = pd.DataFrame(results,columns = columns)
            for h in range(self.nsignals):
                Y_tmp = np.array(y[h])
                Y_tmp = normalize(Y_tmp,norm="l1")
                Hp_pred = np.argmax(Y_tmp,axis=1)
                Y_tmp = np.concatenate((Hp_pred[np.newaxis].T,Y_tmp),axis=1)
                columns = [SignalName[h]+"_Hypno"]+[SignalName[h]+"_"+k for k in list(self.SCORE_DICT.keys())]
                Y_tmp = pd.DataFrame(Y_tmp,columns = columns)
                DF = pd.concat((DF,Y_tmp),axis = 1)
            DF["Measure_date"] = self.generator.currentSignal.metadata["Measure date"]
            DF.to_csv(filepath)
        else:
            columns = ["Times",SignalName[0]+"_Hypno"]+[SignalName[0]+"_"+k for k in list(self.SCORE_DICT.keys())]+["GrayArea"]+["Warning_"+k for k in list(warnings.keys())]
            DF = pd.DataFrame(results,columns = columns)
            DF["Measure_date"] = self.generator.currentSignal.metadata["Measure date"]
            DF.to_csv(filepath)

    def GenerateMultiSamp(self,x,E):
        x = np.array([x]).astype(np.float64)
        if sum(x.sum(axis=1)) != 1:
            x[0,:] = x[0,:]/sum(x[0,:])

        gen = GenMixtSampleFromCatEns(E,x)
        X,Z = gen.generate(2,distribution="Multinomial")
        return X[0,:].tolist()
    
    
def main():
    # logging.basicConfig(level=logging.DEBUG)
    # os.chdir(os.path.join(os.getcwd(),"./api/GrayAreaDL/"))
    yaml_path = str(sys.argv[1])
    
    if len(sys.argv)>2:
        if str(sys.argv[2])!="pred":
            run_pipeline = RunTrain(yaml_path)
            run_pipeline.run_model()
        else:
            run_pipeline = RunPredict(yaml_path)
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
            # if run_pipeline.nfile>1:
            #     num_workers = 2
            #     Parallel(n_jobs=num_workers)(delayed(run_pipeline.PredictToCSV)(file) for file in range(run_pipeline.nfile))
            # else:
            #     run_pipeline.PredictToCSV(0)

            
            for file in range(1,run_pipeline.nfile+1):
                print("-------------------------------------------------------- BEGIN PREDICTION -----------------------------------------------------------------")
                run_pipeline.PredictToCSV(file)


            
    else:
        run_pipeline = RunTrain(yaml_path)
        run_pipeline.run_model()



if __name__ == "__main__":
    main()
