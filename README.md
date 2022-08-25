# Implied_Motion

## Data_Preparation

This folder contains scripts that:
   - Extract v-coco IDs from the v-coco ID API
   - Extract the manually selected subgroup IDs from v-coco that are used for this study specifically
   - Resize the manually selected subgroup v-coco iamges from 425x425 to Im2Flow's input size of 256x256 
   - Extract fMRI betas from trials that used ths subgroup v-coco images
   

## RSA_Analysis
This folder contains scripts that:
   - Conduct first level & second level analysis of both the fMRI data & the network data (flow maps & layers are separated analyses)
   - Statistical Analysis based on the RSA Level 2 Results
