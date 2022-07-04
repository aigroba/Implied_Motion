import os
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise
from sklearn.feature_selection import r_regression
from scipy.stats import pearsonr
from importlib import reload
import matplotlib.pyplot as plt
from collections import defaultdict


SUBJ = ['subj02']
        # 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
for subj in SUBJ:
    layers = ['02/', '05/', '08/', '11/', '14/', '17/', '11/','20/']
    for layer in layers:
        layer_02_path = os.path.join('/home/ana/PycharmProjects/Master/Datasets/NSD/results', subj, 'layers', layer)
        layer_dir = sorted(os.listdir(layer_02_path))
        image_filters = defaultdict(list)

        for img in layer_dir:
            image = np.load(layer_02_path + img)
            shape_1 = len(image[0,:,0,0])
            shape_2 = len(image[0,0,:,0])
            place_holder = np.zeros((shape_1, shape_2*shape_2))
            for i in range(0, shape_1):
                place_holder[i,:]= np.ravel(image[0,i,:,:])
            image_filters[img].append(np.transpose(place_holder))

        correlation_matrix_layer = np.zeros((len(layer_dir), len(layer_dir)))  # flow_components
        # for-loop to calculate cosine similarity
        for idx1 in layer_dir:
            corr_vector = np.zeros((1,len(image_filters[idx1][0][0,:])))  # flow_components
            for idx2 in layer_dir:
                for row in range(0, len(image_filters[idx2][0][0,:])):
                    corr = np.corrcoef(image_filters[idx1][0][:,row], y=image_filters[idx2][0][:,row])
                    # corr = np.corrcoef(image_filters[idx1][0],image_filters[idx2][0])
                    corr_vector[0,row] = corr[0,1]
                correlation_matrix_layer[layer_dir.index(idx1), layer_dir.index(idx2)] = np.nanmean(corr_vector)
            # reset the correlation_vectors
            # corr_vector = []
            print(idx1, '/', len(correlation_matrix_layer), 'Done!')
        #
        # plt.plot(correlation_matrix_layer)
        # plt.show()
        os.makedirs('/home/ana/PycharmProjects/Master/Datasets/NSD/results/' + subj + '/Layer_Correlations/' + layer)
        np.save('/home/ana/PycharmProjects/Master/Datasets/NSD/results/' + subj + '/Layer_Correlations/' + layer +
                                                                                      '/Layer_Correlations_Pearson.npy',
                    correlation_matrix_layer)

