import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import math

layers = ['02', '05', '08', '11', '14', '17', '20']
subjects = ['subj01']
    # ,'subj02','subj03','subj04','subj05','subj06','subj07','subj08']
brain_areas = ['V1', 'V2', 'V3', 'V3A', 'V3B', 'MT', 'V6', 'V7', 'MST', 'LO1', 'LO2']

for subj in subjects:
    correlations_placeholders = np.zeros((len(layers), len(brain_areas)))
    p_value_placeholders = np.zeros((len(layers), len(brain_areas)))
    lower_error_diff = np.zeros((len(layers), len(brain_areas)))
    for layer in layers:
        Layer_Corr_Path = os.path.join('/home/ana/PycharmProjects/Master/Datasets/NSD/results', subj,
                                       'Layer_Correlations', layer, 'Layer_Correlations_Pearson.npy')
        for ROI in brain_areas:
            Brain_Path = os.path.join('/home/ana/PycharmProjects/Master/Datasets/NSD/results',subj,'fMRI_Correlations')
            fMRI_Corr = np.load(Brain_Path +'/Correlations_' + ROI +'.npy')
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
    # fig.axes.set_ylim(0,0.3)
    subject_DataFrame.iloc[0].plot(ax = ax[0,0], kind ='bar', title = subject_DataFrame.index[0], ylabel= 'Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[0,:])
    subject_DataFrame.iloc[1].plot(ax= ax[0,1], kind='bar', title=subject_DataFrame.index[1], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[1,:])
    subject_DataFrame.iloc[2].plot(ax= ax[0,2], kind='bar', title=subject_DataFrame.index[2], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[2,:])
    subject_DataFrame.iloc[3].plot(ax= ax[0,3], kind='bar', title=subject_DataFrame.index[3], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[3,:])
    subject_DataFrame.iloc[4].plot(ax= ax[1,0], kind='bar', title=subject_DataFrame.index[4], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[4,:])
    subject_DataFrame.iloc[5].plot(ax= ax[1,1], kind='bar', title=subject_DataFrame.index[5], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[5,:])
    subject_DataFrame.iloc[6].plot(ax= ax[1,2], kind='bar', title=subject_DataFrame.index[6], ylabel='Correlation',
                                   ylim = (0, 0.3), yerr = lower_error_diff[6,:])

    fig2, ax2 = plt.subplots(3,4)
    fig.suptitle('Correlations displayed per Region')
    subject_DataFrame_T.iloc[0].plot(ax=ax2[0, 0], kind='bar', title=subject_DataFrame_T.index[0], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,0])
    subject_DataFrame_T.iloc[1].plot(ax=ax2[0, 1], kind='bar', title=subject_DataFrame_T.index[1], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,1])
    subject_DataFrame_T.iloc[2].plot(ax=ax2[0, 2], kind='bar', title=subject_DataFrame_T.index[2], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,2])
    subject_DataFrame_T.iloc[3].plot(ax=ax2[0, 3], kind='bar', title=subject_DataFrame_T.index[3], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,3])
    subject_DataFrame_T.iloc[4].plot(ax=ax2[1, 0], kind='bar', title=subject_DataFrame_T.index[4], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,4])
    subject_DataFrame_T.iloc[5].plot(ax=ax2[1, 1], kind='bar', title=subject_DataFrame_T.index[5], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,5])
    subject_DataFrame_T.iloc[6].plot(ax=ax2[1, 2], kind='bar', title=subject_DataFrame_T.index[6], ylabel='Correlation',
                                   ylim=(0, 0.3), yerr = lower_error_diff[:,6])
    subject_DataFrame_T.iloc[7].plot(ax=ax2[1, 3], kind='bar', title=subject_DataFrame_T.index[7], ylabel='Correlation',
                                     ylim=(0, 0.3), yerr = lower_error_diff[:,7])
    subject_DataFrame_T.iloc[8].plot(ax=ax2[2, 0], kind='bar', title=subject_DataFrame_T.index[8], ylabel='Correlation',
                                     ylim=(0, 0.3), yerr = lower_error_diff[:,8])
    subject_DataFrame_T.iloc[9].plot(ax=ax2[2, 1], kind='bar', title=subject_DataFrame_T.index[9], ylabel='Correlation',
                                     ylim=(0, 0.3), yerr = lower_error_diff[:,9])
    subject_DataFrame_T.iloc[10].plot(ax=ax2[2, 2], kind='bar', title=subject_DataFrame_T.index[10], ylabel='Correlation',
                                     ylim=(0, 0.3), yerr = lower_error_diff[:,10])
    plt.show()

    subject_DataFrame.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/subj01/results/Level_2_Correlations.csv')
    subject_DataFrame_p_value.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/subj01/results/Level_2_p_values.csv')



# # Flow_Corr_2 = np.load('/home/ana/PycharmProjects/Master/Datasets/NSD/results/subj01/Layer_Correlations/14 /Layer_Correlations.npy')
# fMRI_V1 = np.load('/home/ana/PycharmProjects/Master/Datasets/NSD/results/subj01/fMRI_Correlations/Correlations_MST.npy')
#
# Flow_flat = np.matrix.flatten(Flow_Corr)
# # Flow_Corr_2_flat = np.matrix.flatten(Flow_Corr_2)
# fMRI_flat = np.matrix.flatten(fMRI_V1)
#
# corr = stats.spearmanr(Flow_flat, fMRI_flat, axis=None)
# # corr_per = stats.permutation_test((Flow_flat, fMRI_flat), statistic=stats.spearmanr, permutation_type='samples', batch=10, n_resamples=500)
# # corr2 = stats.spearmanr(Flow_flat, Flow_Corr_2_flat)
#
# print(corr)
# # print(corr2)
# # print(corr_per)

