# BanknotesRecognition
<br>

## Overview of How the Proposed Method Works
Given input image of Renminbi (RMB) bill, first, the program will perform pre-processing: (i) cropping the bill only (black background is removed), and (ii) taking 4 Potential RoIs per two-sides (front-back) RMB bill. Then, given 4 Potential RoIs per each two-sides RMB bill, BPN (Block-wise Prediction Networks) as the region proposer will locate the ROI of bank serial number. The ROI will be separated into individual characters, and finally each individual character will be classified using CNN classifier.
![alt text](https://github.com/ardianumam/BanknotesRecognition/blob/master/Flowchart_of_proposed_system.jpg "Flowhart of the proposed method")

## Dataset
Dataset
* Number of images: 2,400 two sides images from 1,200 Renminbi bills
* Ground truth label: provied in CSV files with 3 columns (1: identifier of the corresponding image filename,2: serial number (0 means no serial number), 3: length of serial number)

## Cite the paper here
```
@inproceedings{umam2018BPN,
  title={A Light Deep Learning Based Method for Bank Serial Number Recognition},
  author={Umam, Ardian and Chuang, Jen-Hui and Li, Dong-Lin},
  booktitle={IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  year={2018}
}
```
The paper can be accessed [here](#) (will be updated after conference day)
