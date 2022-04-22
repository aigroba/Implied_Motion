# Implied_Motion

## Data_Preparation
This folder contains scripts that:
    - extract v-coco IDs from the v-coco ID API
    - extract the manually selected subgroup IDs from v-coco that are used for this study specifically
    - resize the manually selected subgroup v-coco iamges from 425x425 to Im2Flow's input size of 256x256 
    - extract fMRI betas from trials that used ths subgroup v-coco images
    
    
filter_v_coco:
  - extracts v-coco IDs from the v-coco ID API
  - matches NSD IDs to the v-coco IDs & preserves the subject presentation order
  - creates csv files that contain the index matches between v-coco index/ NSD index
re_scale_img:
  - uses manual v-coco subgroup & resizes them to the appropriate size
  - creates new directory & stores resized images there
data:
  - that's mostly your code with my v-coco filter slapped onto it
  - basically, only the betas associated with the manually selected v-coco images are extracted

The scripts are meant to be run in the order 'filter_v_coco','re_scale_img','data', since they rely on the information of the previous scripts.

## RSA_Analysis
This folder contains scripts that calculate first level RSMs of both the fMRI data & the flow maps.
