import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    SUBJ = ['subj01','subj02','subj03', 'subj04', 
            'subj05', 'subj06', 'subj07', 'subj08']
    save_dir = Path('/path/to/Datasets/NSD/results')
    for subj in SUBJ:
        layers = ['02/', '05/', '08/', '11/', 
                  '14/', '17/', '11/','20/']
        for layer in layers:
            layer_path = save_dir / subj / 'layers' / layer
            layer_dir = sorted(layer_path.glob('*'))
            image_filters = defaultdict(list)

            for img in layer_dir:
                image = np.load(img)
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
                print(idx1, '/', len(correlation_matrix_layer), 'Done!')

            # plt.plot(correlation_matrix_layer)
            # plt.show()
            (save_dir / subj / 'Layer_Correlations' / layer).mkdir(parents=True, exist_ok=True)
            np.save(save_dir / subj / 'Layer_Correlations' / layer / 'Layer_Correlations_Pearson.npy',
                    correlation_matrix_layer)

if __name__ == "__main__":
    main()