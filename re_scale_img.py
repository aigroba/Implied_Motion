import cv2 as cv
import os

# resize the v-coco images selected for this study so that they are the proper input size for Im2Flow

def rescale_study_img(subj):
    Image_Path = os.path.join('/home/ana/PycharmProjects/Master/Datasets/NSD/results', subj, 'Images_425')
    os.mkdir('/home/ana/PycharmProjects/Master/Datasets/NSD/results/' + subj + '/Images_256')
    New_Image_Path = os.path.join('/home/ana/PycharmProjects/Master/Datasets/NSD/results', subj, 'Images_256')
    img_list = os.listdir(Image_Path)
    for img in img_list:
        n = cv.imread(img, cv.IMREAD_UNCHANGED)
        resize = cv.resize(n, (256, 256), interpolation=cv.INTER_AREA)
        cv.imwrite(New_Image_Path, resize)


def main():
    subj = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
    for subj in subj:
        rescale_study_img(subj)


if __name__ == "__main__":
    main()