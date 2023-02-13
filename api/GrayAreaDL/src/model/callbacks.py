from tensorflow import keras
import seaborn as sns
import numpy as np
import os
os.environ["QT_LOGGING_RULES"] = '*.debug=false'
from matplotlib import pyplot as plt

def generate_callbacks(log_dir, patience, path_weights, metric_monitored="loss") -> list:
    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True,
                                             histogram_freq=0,  # How often to log histogram visualizations
                                             embeddings_freq=0,  # How often to log embedding visualizations
                                             update_freq="epoch")

    esCallBack = keras.callbacks.EarlyStopping(monitor=metric_monitored,
                                               min_delta=0,
                                               patience=patience,
                                               verbose=0, mode='auto', restore_best_weights=True)

    mcCallBack = keras.callbacks.ModelCheckpoint(path_weights, monitor=metric_monitored,
                                                 save_weights_only=True, save_freq='epoch',
                                                 save_best_only=True, mode='auto', period=1)
    return [tbCallBack, esCallBack, mcCallBack]


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self,datagenerator, model_name, file_path,i,batch=1):
        self.batch = batch
        self.model_name = model_name
        self.file_path = file_path
        self.i = i
        self.path_image = os.path.join(self.file_path,self.model_name+"_recons")
        os.makedirs(self.path_image, exist_ok=True)
        self.path_map = os.path.join(self.file_path, self.model_name + "_errormap")
        os.makedirs(self.path_map, exist_ok=True)
        self.data = datagenerator.data
        self.plot_result = Display(datagenerator.data,
                              path_to_save="G:\\BEPS-IA\\bepsia\\figures\\datagenerator",
                              date=datagenerator.data.get_date(str(i)))
        self.K = len(datagenerator.data.weather_variable)
        self.T = datagenerator.data.time_window+1
        self.x_test = datagenerator.__getitem__(self.batch)
        indexes = datagenerator.indexes[self.batch * datagenerator.batch_size:(self.batch + 1) * datagenerator.batch_size]
        self.X_true = datagenerator.data.load_images(indexes)[i].get_weather_dict()['Weather_map']
        self.lake_name = datagenerator.data.metadata['data_lake_name'][int(indexes[i])]

    def on_epoch_end(self, epoch, logs={}):
        x_pred = self.model.predict(self.x_test)
        x_rmse = np.sqrt(np.mean((self.x_test-x_pred)**2, axis=(0)))
        h = 0
        for k in range(self.K):
            for T in range(self.T):
                heat_map = x_rmse[:,:,h]
                fig = plt.figure()
                ax = sns.heatmap(heat_map, vmin=0, vmax=0.5, cbar_kws={'label': 'RMSE'})
                ax.set_title(f'Epoch {epoch}')

                plt.savefig(os.path.join(self.path_map,
                                         f"heatmap_{self.data.weather_variable[k]}_{T}_Epoch_{epoch}.png"))
                plt.close(fig)

                test_tmp = self.x_test[self.i, :, :, h]
                tmp = x_pred[self.i, :, :, h]

                fig, axn = plt.subplots(1, 3, figsize=(12, 3))
                ax = self.plot_result.heatmap(data=self.X_true[k,T,:,:],
                                         only=False,
                                         ax=axn[0])
                ax.set_title(f'True {self.data.weather_variable[k]} {self.data.get_date(str(self.i))} {self.lake_name}')
                ax = self.plot_result.heatmap(data=test_tmp,
                                         only=False,
                                         ax=axn[1])
                ax.set_title(f'Reshape ')
                ax = self.plot_result.heatmap(data=tmp,
                                         only=False,
                                         ax=axn[2])
                ax.set_title(f'Predict')

                plt.savefig(os.path.join(self.path_image,
                                         f"{self.data.weather_variable[k]}_{T}_epoch_{epoch}_plot.png"))
                plt.close(fig)
                h=h+1