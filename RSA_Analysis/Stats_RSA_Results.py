from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def group_results_layers(Filepath_Data, subjects, layers, brain_areas):
    Grouped_Results = np.zeros((len(subjects), len(layers), len(brain_areas)))
    for SUBJ in subjects:
        load_path = Filepath_Data / 'results' / SUBJ /'results' / 'Level_2_Correlations_lower_triangle.csv'
        tmp = pd.read(load_path)
        for label in brain_areas:
            Grouped_Results[(subjects.index(SUBJ)), :, (brain_areas.index(label))] = tmp[label].to_numpy()
    return Grouped_Results


def group_results_Flow(Filepath_Data, subjects, Flow_Maps, brain_areas):
    Grouped_Results_Flow = np.zeros((len(subjects), len(Flow_Maps), len(brain_areas)))
    for SUBJ in subjects:
        load_path = Filepath_Data / 'results' / SUBJ / 'results' / 'Level_2_Correlations_lower_triangle_Flow.csv'
        tmp = pd.read(load_path)
        for label in brain_areas:
            Grouped_Results_Flow[(subjects.index(SUBJ)), :, (brain_areas.index(label))] = tmp[label].to_numpy()
    return Grouped_Results_Flow


def RSA_stats(Grouped_Results, wilcoxon_method, wilcoxon_alternative):
    group_fisher_tr = np.arctanh(Grouped_Results)
    wilc_layers = np.zeros((len(group_fisher_tr[0, :, 0]), 2))
    results_wilc_ROIS = np.zeros((len(group_fisher_tr[0, 0, :]), 2))
    wilc_ROIxL = np.zeros((len(group_fisher_tr[0, :, 0]), 
                                         len(group_fisher_tr[0, 0, :])))
    pval_ROIxL = np.zeros((len(group_fisher_tr[0, :, 0]), 
                                len(group_fisher_tr[0, 0, :])))


    for layer in range(len(group_fisher_tr[0, :, 0])):
        wilc_layers[layer] = stats.wilcoxon(group_fisher_tr[:, layer, 0],
                                                             method= wilcoxon_method,
                                                             alternative= wilcoxon_alternative)
        for ROI in range(len(group_fisher_tr[0, 0, :])):
            [wilc_ROIxL[layer, ROI], pval_ROIxL[layer, ROI]] = stats.wilcoxon(group_fisher_tr[:, layer, ROI], 
                                                                              method= wilcoxon_method,
                                                                              alternative= wilcoxon_alternative)

            
    for ROI in range(len(group_fisher_tr[0, 0, :])):
        results_wilc_ROIS[ROI] = stats.wilcoxon(group_fisher_tr[:, 0, ROI], 
                                                         method= wilcoxon_method,
                                                         alternative=wilcoxon_alternative)
    wilc_corr = sm.stats.fdrcorrection(
        np.concatenate((wilc_layers[:, 1], results_wilc_ROIS[:, 1])), alpha=0.05,
        method='negcorr')
    wilc_corr_ROIxL = sm.stats.fdrcorrection(pval_ROIxL.ravel(), alpha=0.05,
                                                                      method='negcorr')
    return (wilc_ROIxL, results_wilc_ROIS, wilc_layers, 
            wilc_corr_ROIxL, wilc_corr)


def Descriptives(Grouped_Results, save, 
                 NSD_PATH: Path = Path('/path/to//Datasets/NSD')):
    ### to save these tables, set save to an integer value of 1
    Desc_Mean = pd.DataFrame(np.nanmean(Grouped_Results, axis=0))
    Desc_Min = pd.DataFrame(np.min(Grouped_Results, axis=0))
    Desc_Max = pd.DataFrame(np.max(Grouped_Results, axis=0))
    Desc_SEM = pd.DataFrame(stats.sem(Grouped_Results, axis=0))
    if save == 1:
        save_path = NSD_PATH / 'results' / 'Descriptives'
        save_path.mkdir(parents=True, exist_ok=True)
        Desc_Mean.to_csv(save_path / 'Desc_Mean_Flow.csv')
        Desc_Min.to_csv(save_path / 'Desc_Min_Flow.csv')
        Desc_Max.to_csv(save_path / 'Desc_Max_Flow.csv')
        Desc_SEM.to_csv(save_path / 'Desc_SEM_Flow.csv')
    return Desc_Mean, Desc_Min, Desc_Max, Desc_SEM

def main():
    layers = ['02', '05', '08', '11', '14', '17', '20', 
              '25', '30', '35', '40', '44', '48', '52']
    subjects = ['subj01', 'subj02', 'subj03', 'subj04', 
                'subj05', 'subj06', 'subj07', 'subj08']  #
    brain_areas = ['V1', 'V2', 'V3', 'V3A', 'V3B', 
                   'MT', 'V6', 'V7', 'MST']
    Flow_Maps = ['Correlation_vectors_2', 'Correlation_vectors_mgt']
    NSD_PATH = Path('/path/to/Datasets/NSD')
    wilcoxon_method = 'exact'
    wilcoxon_alternative = 'greater'

    Grouped_Results_Layers = group_results_layers(NSD_PATH, subjects, layers, brain_areas)
    Grouped_Results_Flow = group_results_Flow(NSD_PATH, subjects, Flow_Maps, brain_areas)
    Wilcoxon_Results_Layers = RSA_stats(Grouped_Results_Layers, wilcoxon_method, wilcoxon_alternative)
    Wilcoxon_Results_Flow = RSA_stats(Grouped_Results_Flow, wilcoxon_method, wilcoxon_alternative)

    Desc_Layers = Descriptives(Grouped_Results_Layers, 0)
    Desc_Flow = Descriptives(Grouped_Results_Flow, 0)


