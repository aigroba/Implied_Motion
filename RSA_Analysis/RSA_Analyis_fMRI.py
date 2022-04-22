import os
import numpy as np
from scipy import stats


def RDM_fMRI(subj, brain_regions):
    for SUBJ in subj:
        for region in brain_regions:
            # load in fMRI_data for the given brain region
            tmp_path = os.path.join('/Datasets/NSD/results', SUBJ, 'betas/HCP_MMP1',
                                    region + '.npy')
            brain_area_data = np.load(tmp_path)
            correlation_matrix = np.zeros(
                (len(brain_area_data), len(brain_area_data)))  # placeholder correlation matrix

            # calculate Pearson correlation between averaged image trials
            for idx1 in range(len(brain_area_data)):
                corr_vector = [] # placeholder array
                for idx2 in range(len(brain_area_data)):
                    corr = stats.pearsonr(brain_area_data[idx1], brain_area_data[idx2])
                    corr_vector.append(corr[0]) # only append the correlation value & exclude the p-value
                correlation_matrix[idx1, :] = corr_vector
                corr_vector = [] # reset placeholder array for next correlation iteration
                print((idx1 + 1), '/', len(brain_area_data))

            np.save(
                '/home/ana/PycharmProjects/Master/Datasets/NSD/results/' + SUBJ + '/fMRI_Correlations/Correlations_' + region + '.npy',
                correlation_matrix)


def main():
    brain_regions = ['V1', 'V2', 'V3', 'V3A', 'V3B', 'V6', 'V7', 'MT', 'MST', 'LO1', 'LO2']
    subj = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
    RDM_fMRI(subj, brain_regions)


if __name__ == "__main__":
    main()
