import os
import numpy as np
from collections import defaultdict
import pandas as pd
import csv

NSD_PATH = '/Datasets/NSD'
v_coco_path = '/v-coco/data/splits/vcoco_all.ids'

def get_img_ids(subj):
    responses_path = os.path.join(NSD_PATH, "responses",
                                  subj, "responses.tsv")

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


def get_mscoco(subj):
    stim_info_path = os.path.join(NSD_PATH, 'stimuli/nsd_stim_info_merged.csv')
    nsd_ids = get_img_ids(subj)

    df = pd.read_csv(stim_info_path)
    coco_ids = df["cocoId"].to_numpy()
    coco_ids_subj = coco_ids[np.array([*nsd_ids.keys()]) - 1]
    # convert variable to list for later comparisons
    coco_ids = coco_ids.tolist()
    return coco_ids_subj, coco_ids


def get_vcoco_list(v_coco_path):
    # load the v-coco Ids
    vcoco_id = open(v_coco_path)
    vcoco_list = list(map(int, vcoco_id.read().split()))
    return vcoco_list


def order_vcoco(coco_ids_subj, vcoco_list):
    v_coco_ids = [] # placeholder list of v_coco_ids, but in presentation order

    # compares cocoIDs of subject and the vcoco_list & appends them in correct order to a new list
    for i in coco_ids_subj:
        for n in vcoco_list:
            if n == i:
                v_coco_ids.append(i)
    return v_coco_ids


def match_vcoco_nsd(v_coco_ids, coco_ids):
    v_coco_nsd_id = [] # placeholder list of nsd_ids associated with the vcoco Ids
    # converts list of vcoco_Ids to the associated NSD_Ids
    for i in v_coco_ids:
        for n in coco_ids:
            if i == n:
                v_coco_nsd_id.append((coco_ids.index(n)) + 1)
    return v_coco_nsd_id


def get_vcoco_nsd(subj):
    # final function used in data.py that contains all previous functions
    # this function maintains original NSD presentation order, but only contains the v-coco images
    nsd_ids = get_img_ids(subj)
    coco_ids_subj, coco_ids = get_mscoco(subj)
    v_coco_list = get_vcoco_list(v_coco_path)
    v_coco_ids = order_vcoco(coco_ids_subj, v_coco_list)
    v_coco_nsd_id = match_vcoco_nsd(v_coco_ids, coco_ids)

    true_list = defaultdict(list)  # final variable with vcoco_IDs in same data structure as in data.py
    # this ensures that the presentation order of NSD Ids is kept intact
    for idx in nsd_ids:
        for i in v_coco_nsd_id:
            if idx == i:
                true_list[i].append(nsd_ids[i])
    return true_list

def main():
    # this part of the script extracts the index info and saves it as csv files. It is not needed for data.py
    subj = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
    NSD_PATH = '/Datasets/NSD'

    for subj in subj:
        v_coco_list = get_vcoco_list(v_coco_path)
        coco_ids_subj, coco_ids = get_mscoco(subj)
        v_coco_ids = order_vcoco(coco_ids_subj, v_coco_list)
        v_coco_nsd_id = match_vcoco_nsd(v_coco_ids, coco_ids)
        # true_list = get_vcoco_nsd(subj)

        # all the paths we will need
        save_path = os.path.join(NSD_PATH, 'results', subj, 'mscoco')
        save_file = os.path.join(save_path, "v_coco_info.csv")
        save_file_2 = os.path.join(save_path, "v_coco_list.csv")

        # this writes the list into csv files
        header = ['nsd_Id', "v_coco_id", "coco_Id"]
        Check = [x in v_coco_list for x in coco_ids_subj]
        data = zip(Check, coco_ids_subj)
        data2 = zip(v_coco_nsd_id, v_coco_ids)

        with open(save_file, "w") as f:
            csv_out = csv.writer(f)
            csv_out.writerow(header[1:])
            csv_out.writerows(data)
            f.close()
        with open(save_file_2, "w") as f:
            csv_out = csv.writer(f)
            csv_out.writerow(header[:2])
            csv_out.writerows(data2)
            f.close()
        print(subj, 'finished')


if __name__ == "__main__":
    main()