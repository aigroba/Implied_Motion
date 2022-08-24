import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def group_results_layers(Filepath_Data, subjects, layers, brain_areas):
    Grouped_Results = np.zeros((len(subjects), len(layers), len(brain_areas)))
    for SUBJ in subjects:
        load_path = os.path.join(Filepath_Data, SUBJ,
                                 'results/Level_2_Correlations_lower_triangle.csv')
        tmp = pd.read(load_path)
        for label in brain_areas:
            Grouped_Results[(subjects.index(SUBJ)), :, (brain_areas.index(label))] = tmp[label].to_numpy()
    return Grouped_Results


def group_results_Flow(Filepath_Data, subjects, Flow_Maps, brain_areas):
    Grouped_Results_Flow = np.zeros((len(subjects), len(Flow_Maps), len(brain_areas)))
    for SUBJ in subjects:
        load_path = os.path.join(Filepath_Data, SUBJ,
                                     'results/Level_2_Correlations_lower_triangle_Flow.csv')
        tmp = pd.read(load_path)
        for label in brain_areas:
            Grouped_Results_Flow[(subjects.index(SUBJ)), :, (brain_areas.index(label))] = tmp[label].to_numpy()
    return Grouped_Results_Flow


def RSA_stats(Grouped_Results, wilcoxon_method, wilcoxon_alternative):
    Grouped_Results_Fisher_transformed = np.arctanh(Grouped_Results)
    Results_Wilcoxon_test_layers = np.zeros((len(Grouped_Results_Fisher_transformed[0, :, 0]), 2))
    Results_Wilcoxon_test_ROIxLayers = np.zeros((len(Grouped_Results_Fisher_transformed[0, :, 0]), len(Grouped_Results_Fisher_transformed[0, 0, :])))
    p_values_ROIxLayers = np.zeros((len(Grouped_Results_Fisher_transformed[0, :, 0]), len(Grouped_Results_Fisher_transformed[0, 0, :])))
    Results_Wilcoxon_test_ROIS = np.zeros((len(Grouped_Results_Fisher_transformed[0, 0, :]), 2))

    for layer in range(len(Grouped_Results_Fisher_transformed[0, :, 0])):
        Results_Wilcoxon_test_layers[layer] = stats.wilcoxon(Grouped_Results_Fisher_transformed[:, layer, 0],
                                                             method= wilcoxon_method,
                                                             alternative= wilcoxon_alternative)
        for ROI in range(len(Grouped_Results_Fisher_transformed[0, 0, :])):
            [Results_Wilcoxon_test_ROIxLayers[layer, ROI], p_values_ROIxLayers[layer, ROI]] = stats.wilcoxon(
                Grouped_Results_Fisher_transformed[:, layer, ROI], method= wilcoxon_method,
                alternative= wilcoxon_alternative)
    for ROI in range(len(Grouped_Results_Fisher_transformed[0, 0, :])):
        Results_Wilcoxon_test_ROIS[ROI] = stats.wilcoxon(Grouped_Results_Fisher_transformed[:, 0, ROI], method= wilcoxon_method,
                                                         alternative=wilcoxon_alternative)
    Results_Wilcoxon_test_Corrected = sm.stats.fdrcorrection(
        np.concatenate((Results_Wilcoxon_test_layers[:, 1], Results_Wilcoxon_test_ROIS[:, 1])), alpha=0.05,
        method='negcorr')
    Results_Wilcoxon_test_Corrected_ROIxLayers = sm.stats.fdrcorrection(p_values_ROIxLayers.ravel(), alpha=0.05,
                                                                      method='negcorr')
    return Results_Wilcoxon_test_ROIxLayers, Results_Wilcoxon_test_ROIS, Results_Wilcoxon_test_layers, Results_Wilcoxon_test_Corrected_ROIxLayers, Results_Wilcoxon_test_Corrected


def Descriptives(Grouped_Results, save):
    ### to save these tables, set save to an integer value of 1
    Desc_Mean = pd.DataFrame(np.nanmean(Grouped_Results, axis=0))
    Desc_Min = pd.DataFrame(np.min(Grouped_Results, axis=0))
    Desc_Max = pd.DataFrame(np.max(Grouped_Results, axis=0))
    Desc_SEM = pd.DataFrame(stats.sem(Grouped_Results, axis=0))
    if save == 1:
        if not os.path.isdir('/home/ana/PycharmProjects/Master/Datasets/NSD/results/Descriptives/'):
            os.makedirs('/home/ana/PycharmProjects/Master/Datasets/NSD/results/Descriptives/')
        Desc_Mean.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/RDMs/Desc_Mean_Flow.csv')
        Desc_Min.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/RDMs/Desc_Min_Flow.csv')
        Desc_Max.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/RDMs/Desc_Max_Flow.csv')
        Desc_SEM.to_csv('/home/ana/PycharmProjects/Master/Datasets/NSD/results/RDMs/Desc_SEM_Flow.csv')
    return Desc_Mean, Desc_Min, Desc_Max, Desc_SEM

def main():
    layers = ['02', '05', '08', '11', '14', '17', '20', '25', '30', '35', '40', '44', '48', '52']
    subjects = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']  #
    brain_areas = ['V1', 'V2', 'V3', 'V3A', 'V3B', 'MT', 'V6', 'V7', 'MST']
    Flow_Maps = ['Correlation_vectors_2', 'Correlation_vectors_mgt']
    Load_Path = '/home/ana/PycharmProjects/Master/Datasets/NSD/results'
    wilcoxon_method = 'exact'
    wilcoxon_alternative = 'greater'

    Grouped_Results_Layers = group_results_layers(Load_Path, subjects, layers, brain_areas)
    Grouped_Results_Flow = group_results_Flow(Load_Path, subjects, Flow_Maps, brain_areas)
    Wilcoxon_Results_Layers = RSA_stats(Grouped_Results_Layers, wilcoxon_method, wilcoxon_alternative)
    Wilcoxon_Results_Flow = RSA_stats(Grouped_Results_Flow, wilcoxon_method, wilcoxon_alternative)

    Desc_Layers = Descriptives(Grouped_Results_Layers, 0)
    Desc_Flow = Descriptives(Grouped_Results_Flow, 0)


