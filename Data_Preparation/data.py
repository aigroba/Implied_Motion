import csv
import glob
import cv2
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy.stats import zscore
from filter_v_coco import get_vcoco_nsd



def get_img_ids(subj,NSD_PATH: Path = Path('/path/to/Datasets/NSD')):
    responses_path = NSD_PATH / "responses" / subj / "responses.tsv"
    df = pd.read_csv(responses_path, sep="\t")

    # only consider existing sessions
    num_ses = df['SESSION'].max() - 3
    df = df.loc[df['SESSION'] <= num_ses]

    # IDS start at 1 and go up to 73000
    img_ids_sorted = df["73KID"].tolist()
    img_ids_unique = defaultdict(list)

    for idx, val in enumerate(img_ids_sorted):
        img_ids_unique[val].append(idx)

    return img_ids_unique


def get_study_vcoco(subj, NSD_PATH: Path = Path('/path/to/Datasets/NSD')):
    # since I manually selected a sub-set of images in v-coco, we can't just use the v-coco-id list from the original
    # repository. This reads in the manually selected image labels (NSD indexing) directly into the script and
    # filters for those images within the v-coco-list
    vcoco_img_path = NSD_PATH / 'results' / subj / 'Images_256'
    # vcoco_img_path.chdir()
    v_img_list = vcoco_img_path.glob('*.png')
    v_img_idx = []
    for i in v_img_list:
        v_img_idx.append((int(i.replace('.png','')))+ 1)

    true_list = get_vcoco_nsd(subj)
    v_img = defaultdict(list)
    for idx, val in true_list.items():
        for idx2 in v_img_idx:
            if idx == idx2:
                v_img[idx].append(val)
    return v_img


def get_roi_masks(subj, data_format, atlas,
                  NSD_PATH: Path = Path('/path/to/Datasets/NSD')):
    
    atlas_path = NSD_PATH / "rois" / subj / data_format / f"{atlas}.nii.gz"
    atlas_file = nib.load(atlas_path).get_fdata()
    atlas_file = np.transpose(atlas_file, [2, 1, 0])

    labels_path = NSD_PATH / "rois" / "labels" / f"{atlas}.mgz.ctab"
    df = pd.read_csv(labels_path, delim_whitespace=True)
    labels_file = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

    roi_masks = {}
    for name, digit in labels_file.items():
        roi_masks[name] = atlas_file == digit

    return roi_masks


def get_betas(subj, data_format, data_type, roi_masks,
              NSD_PATH: Path = Path('/path/to/Datasets/NSD')):

    betas_path = NSD_PATH / "betas" / subj / data_format / data_type
    session_files = sorted(glob.glob(betas_path / "*.hdf5"))
    roi_betas = defaultdict(list)

    for session_file in tqdm(session_files):
        with h5py.File(session_file, "r") as f:
            betas_ses = np.array(f["betas"])

        for name, mask in roi_masks.items():
            roi_betas_ses = betas_ses[:, mask].astype(np.float32) / 300
            roi_betas[name].append(zscore(roi_betas_ses, axis=0, ddof=1))

    for name, roi_betas_list in roi_betas.items():
        roi_betas[name] = np.concatenate(roi_betas_list, 0)

    return roi_betas


def avg_trials(betas, img_ids):
    betas_avg = defaultdict(list)
    reps_var = defaultdict(list)

    num_reps = []

    for rep_ids in img_ids.values():
        num_reps.append(len(rep_ids))

        for roi_name, roi_betas in betas.items():
            trials = roi_betas[np.array(rep_ids)]

            roi_betas_avg = np.mean(trials, 0)
            betas_avg[roi_name].append(roi_betas_avg)

            if len(rep_ids) == 3:
                roi_reps_var = np.var(trials, 0, ddof=1)
                reps_var[roi_name].append(roi_reps_var)

    for name, roi_betas_avg in betas_avg.items():
        betas_avg[name] = np.vstack(roi_betas_avg)

    for name, roi_reps_var in reps_var.items():
        reps_var[name] = np.mean(roi_reps_var, 0)

    return betas_avg, reps_var, num_reps


def save_images(subj, img_ids, size=(320, 320), 
                NSD_PATH: Path = Path('/path/to/Datasets/NSD')):

    stimuli_path = NSD_PATH / "stimuli" / "nsd_stimuli.hdf5"
    for idx, img_id in enumerate(img_ids):
        with h5py.File(stimuli_path, "r") as f:
            image = np.array(f["imgBrick"][img_id - 1])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        save_path = NSD_PATH/ 'results' / subj / "images" / str(size[0])
        save_file = save_path / f"{str(idx + 1).zfill(5)}.png"
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_file, image)


def save_salicon_overlap(subj, img_ids, 
                         NSD_PATH: Path = Path('/path/to/Datasets/NSD')):
    
    train_paths = glob.glob(NSD_PATH / "salicon" / "train" / "*")
    valid_paths = glob.glob(NSD_PATH / "salicon" / "val" / "*")

    img_paths = sorted(train_paths + valid_paths)

    img_names = [img_paths(x).name for x in img_paths]
    img_names = [Path(x).stem for x in img_names]
    img_names = [x.split("_", 2)[-1] for x in img_names]
    salicon_ids = [int(x) for x in img_names]

    stim_info_path = NSD_PATH / "stimuli" / "nsd_stim_info_merged.csv"
    df = pd.read_csv(stim_info_path)
    coco_ids = df["cocoId"].to_numpy()

    coco_ids_subj = coco_ids[np.array([*img_ids.keys()]) - 1]
    salicon = [x in salicon_ids for x in coco_ids_subj]

    header = ["cocoId", "salicon"]
    data = zip(coco_ids_subj, salicon)

    save_path = NSD_PATH / 'results' / subj / "mscoco"
    save_file = save_path /"info.csv"

    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_file, "w") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(header)
        csv_out.writerows(data)


def save_noise_ceiling(subj, data_format, data_type,
                       atlas, roi_masks, num_reps,
                       NSD_PATH: Path = Path('/path/to/Datasets/NSD')):

    noise_path = NSD_PATH / "noise" / subj / data_format / data_type /"ncsnr.nii.gz"
    noise = nib.load(noise_path).get_fdata()
    noise = np.transpose(noise, [2, 1, 0])

    count = Counter(num_reps)
    custom = count[3] / 3 + count[2] / 2 + count[1] / 1
    custom = custom / (count[3] + count[2] + count[1])

    noise = np.sqrt(noise**2 / (noise**2 + custom))

    for name, mask in roi_masks.items():
        save_path = NSD_PATH / 'results' / subj / "noise" / atlas
        save_file = save_path / f"{name}.npy"

        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_file, "wb") as f:
            np.save(f, noise[mask])


def save_betas(subj, atlas, betas, 
               NSD_PATH: Path = Path('/path/to/Datasets/NSD')):

    for roi_name, roi_betas in betas.items():
        save_path = NSD_PATH / 'results' / subj / "betas" / atlas
        save_file = save_path / f"{roi_name}.npy"
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_file, "wb") as f:
            np.save(f, roi_betas)


def save_variance(subj, atlas, variance, 
                  NSD_PATH: Path = Path('/path/to/Datasets/NSD')):
    for roi_name, roi_variance in variance.items():
        save_path = NSD_PATH / 'results' / subj / "variance" / atlas
        save_file = save_path / f"{roi_name}.npy"

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_file, "wb") as f:
            np.save(f, roi_variance)


def main():
    NSD_PATH = Path("path/to/Datasets/NSD")
    data_format = "func1pt8mm"
    data_type = "betas_fithrf_GLMdenoise_RR"
    SUBJS = ['subj01', 'subj02', 'subj03', 'subj04', 
             'subj05', 'subj06', 'subj07', 'subj08']
    
    for subj in SUBJS:
        img_ids = get_img_ids(subj)
        true_list = get_study_vcoco(subj)

        save_images(subj, img_ids)
        save_salicon_overlap(subj, img_ids)
        for atlas in ["streams", "HCP_MMP1", "prf-visualrois"]:
            roi_masks = get_roi_masks(subj, data_format, atlas)

            betas = get_betas(subj, data_format, data_type, roi_masks)
            betas_avg, variance, num_reps = avg_trials(betas, true_list)

            save_noise_ceiling(subj, data_format, data_type,
                               atlas, roi_masks, num_reps)

            save_betas(subj, atlas, betas_avg)
            save_variance(subj, atlas, variance)

if __name__ == "__main__":
    main()
