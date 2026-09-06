from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import math

def main():
    layers = ['02', '05', '08', '11', '14', '17', '20']
    subjects = ['subj01','subj02','subj03','subj04',
                'subj05','subj06','subj07','subj08']
    brain_areas = ['V1', 'V2', 'V3', 'V3A', 'V3B', 
                'MT', 'V6', 'V7', 'MST', 'LO1', 'LO2']

    NSD_PATH = Path("path/to/Datasets/NSD/")
    for subj in subjects:
        correlations_placeholders = np.zeros((len(layers), len(brain_areas)))
        p_value_placeholders = np.zeros((len(layers), len(brain_areas)))
        lower_error_diff = np.zeros((len(layers), len(brain_areas)))
        for layer in layers:
            Layer_Corr_Path = NSD_PATH / 'results' / subj / 'Layer_Correlations' / layer / 'Layer_Correlations_Pearson.npy'
            for ROI in brain_areas:
                Brain_Path = NSD_PATH / 'results' / subj / 'fMRI_Correlations' / f'Correlations_{ROI}.npy'
                fMRI_Corr = np.load(Brain_Path)
                layer_Corr = np.load(Layer_Corr_Path)

                fMRI_Corr_Flat = np.matrix.flatten(fMRI_Corr)
                layer_Corr_Flat = np.matrix.flatten(layer_Corr)

                corr = stats.spearmanr(fMRI_Corr_Flat, layer_Corr_Flat)

                # standard error calculation
                num = len(layer_Corr_Flat)
                stderr = (1 + corr[0]/2)/math.sqrt((num - 3))
                delta = 1.96 * stderr
                lower_error_diff[layers.index(layer), brain_areas.index(ROI)] = math.tanh(math.atanh(corr[0]) - delta)
                #
                correlations_placeholders[layers.index(layer), brain_areas.index(ROI)] = corr[0]
                p_value_placeholders[layers.index(layer), brain_areas.index(ROI)] = corr[1]

        subject_DataFrame = pd.DataFrame(data=correlations_placeholders, index = layers, columns= brain_areas)
        subject_DataFrame_p_value = pd.DataFrame(data=correlations_placeholders, index = layers, columns= brain_areas)
        subject_DataFrame_T = subject_DataFrame.T
        lower_error_diff = correlations_placeholders - lower_error_diff


        fig, ax = plt.subplots(2,4)
        fig.suptitle('Correlations displayed per layer')
        fig.axes.set_ylim(0,0.3)
        ax = ax.flatten()
        for i in range(len(layers)):
            subject_DataFrame.iloc[i].plot(
                ax = ax[i], kind ='bar', 
                title = subject_DataFrame.index[i],
                ylabel= 'Correlation', ylim = (0, 0.3), 
                yerr = lower_error_diff[i,:]
                )

        fig2, ax2 = plt.subplots(3,4)
        fig2.suptitle('Correlations displayed per Region')
        ax = ax.flatten()
        for i in range(len(brain_areas)):
            subject_DataFrame_T.iloc[i].plot(
                ax = ax2.flatten()[i], kind ='bar', 
                title = subject_DataFrame_T.index[i],
                ylabel= 'Correlation', ylim = (0, 0.3), 
                yerr = lower_error_diff[:,i]
                )
        plt.show()

        subject_DataFrame.to_csv(NSD_PATH / 'results' / subj / 'Level_2_Correlations.csv')
        subject_DataFrame_p_value.to_csv(NSD_PATH / 'results' / subj / 'Level_2_p_values.csv')

