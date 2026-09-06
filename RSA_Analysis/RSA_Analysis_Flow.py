from pathlib import Path
import imageio
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances


def flow_vectors(subj, NSD_PATH: str = '/Datasets/NSD/'):

    flow_img_vectors_mgt = []
    flow_img_vectors = []
    flow_img_mgt = []

    flow_img_path = NSD_PATH / 'results' / subj / 'flow/'
    flow_dir = sorted((flow_img_path).listdir())

    for img in flow_dir:
        # load in pictures from image directory
        flow_img_idx = int(img.replace('.png', '')) - 1
        Image = imageio.imread(flow_img_path + img)

        # Vector Extraction from pixel RGB value based on Im2Flow's 'Visualize Flow' script
        mgt = Image[:, :, 2] / 255.0 * 30
        sin = Image[:, :, 0] / 255.0 * 2 - 1
        cos = Image[:, :, 1] / 255.0 * 2 - 1

        um = np.ravel(cos * mgt)
        u = np.ravel(cos)
        v = np.ravel(sin)
        vm = np.ravel(sin * mgt)
        wm = np.ravel(mgt)

        # join cosinus & sinus into one array
        vector_mgt = np.transpose(np.vstack((um, vm)))
        vector_2 = np.transpose(np.vstack((u, v)))
        # fill in the dictionaries
        flow_img_vectors_mgt.append(vector_mgt)
        flow_img_vectors.append(vector_2)
        flow_img_mgt.append(wm)

    # return variables: vector components * magnitude, vector components, magnitude
    return flow_img_vectors_mgt, flow_img_vectors, flow_img_mgt


def flow_vectors_RDM(subj, NSD_PATH: str = '/Datasets/NSD/'):
    # vector_mgt: flow_vectors including magnitude
    # flow_components: only cosinus & sinus components
    # mgt: magnitude value only
    vectors_mgt, flow_components, mgt = flow_vectors(subj)

    # placeholder correlation matrices
    correlation_matrix_mgt = np.zeros((len(vectors_mgt), len(vectors_mgt)))  # vector_mgt
    correlation_matrix_2 = np.zeros((len(flow_components), len(flow_components)))  # flow_components
    correlation_matrix_3 = np.zeros((len(mgt), len(mgt)))  # mgt

    # for-loop to calculate cosine similarity
    for idx1 in range(len(correlation_matrix_mgt)):
        # placeholder vectors
        corr_vector_mgt = []  # vector_mgt
        corr_vector_components = []  # flow_components
        corr_mgt = []  # mgt

        for idx2 in range(len(correlation_matrix_mgt)):
            corr = 1 - (paired_cosine_distances(vectors_mgt[idx1], vectors_mgt[idx2]))
            corr_2 = 1 - (paired_cosine_distances(flow_components[idx1], flow_components[idx2]))
            corr_3 = 1 - (paired_cosine_distances(mgt[idx1].reshape(-1, 1), mgt[idx2].reshape(-1, 1)))

            # calculate correlation between images based on average pixel cosine similarity
            corr_vector_mgt.append(np.nanmean(corr))
            corr_vector_components.append(np.nanmean(corr_2))
            corr_mgt.append(np.nanmean(corr_3))

        # append the mean values to the empty correlation matrices
        correlation_matrix_mgt[idx1, :] = corr_vector_mgt
        correlation_matrix_2[idx1, :] = corr_vector_components
        correlation_matrix_3[idx1, :] = corr_mgt
        # reset the correlation_vectors
        corr_vector_mgt = []
        corr_vector_components = []
        corr_mgt = []
        print(idx1 + 1, '/', len(correlation_matrix_mgt), 'Done!')
        np.save(NSD_PATH / 'results' / subj / 'Flow_Correlations' / 'Correlation_vectors_mgt.npy',
                correlation_matrix_mgt)
        np.save(NSD_PATH / 'results' / subj / 'Flow_Correlations' / 'Correlation_vectors_2.npy',
                correlation_matrix_2)
        np.save(NSD_PATH / 'results' / subj / 'Flow_Correlations' / 'Correlation_magnitude.npy',
                correlation_matrix_3)


def main():
    subj = ['subj01', 'subj02', 'subj03', 'subj04', 
            'subj05', 'subj06', 'subj07', 'subj08']
    for subj in subj:
        print('Start calculations for ', subj)
        flow_vectors_RDM(subj)


if __name__ == "__main__":
    main()
