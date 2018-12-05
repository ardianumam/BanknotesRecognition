This repository provides all the dataset with the code used in our paper.

## Overview of How the Proposed Method Works
Given input image of Renminbi (RMB) bill, first, the program will perform pre-processing: (i) background removal, and (ii) taking 4 *Potential RoIs* per two-sides (front-back) RMB bill. Then, given 4 *Potential RoIs* per each two-sides RMB bill, **BPN** (*Block-wise Prediction Networks*) as the region proposer will locate the region of interest (ROI) of bank serial number. The ROI will be separated into individual characters, and finally each individual character will be classified using CNN classifier. (See Figure below)
![alt text](https://github.com/ardianumam/BanknotesRecognition/blob/master/Flowchart_of_proposed_system.jpg "Flowhart of the proposed method")

## Dataset
According the flowchart above, we have two groups of training data: (i) for region proposing, and (ii) for character classification. 
<br>
**(i) For region proposing** (can be downloaded [here](https://drive.google.com/file/d/1_lOXF9w-qYzza2jbmLo9hANubPv2mLRM/view?usp=sharing))
* Consist of 20 positive *Potential RoIs* and 30 negative *Potential RoIs*,
* Bouding box ground truth (for positive *Potential RoIs*) is provided in *txt file* with a format of ![](https://latex.codecogs.com/gif.latex?x_%7Btop-left%7D%2C%20y_%7Btop-left%7D%2C%20x_%7Bbottom-right%7D%2C%20y_%7Bbottom-right%7D). 

**(ii) For character classification** (can be downloaded [here](https://drive.google.com/file/d/1DGG10qL5vw8_9zS4l5sPZ58Y8de5ijzG/view?usp=sharing))
* Consist of 34 classes ("I" and "O" are treated in the same class with "1" and "0", respectively),
* Each class has 20 training images.

The full original data, raw RMB bill images scanned from the machine, with their serial number label can be downloaded [here](https://drive.google.com/file/d/1QCLGAhL34i9qIHfbVbG9o6zbKUKEqtOi/view?usp=sharing), with a specification as follows.  
* Number of images: 2,400 two sides images from 1,200 Renminbi bills,
* Ground truth label: provided in CSV files with 3 columns (1: identifier of the corresponding image filename, 2: serial number (0 means having no serial number), 3: length of serial number).

## How to Run the Code 
The code provided here is only for testing version (cannot be used for training), in a single '*.py*' file: ```RmbBanknotesRecognition.py```. To run the code:
* First, download the trained model & sample input images [here](https://drive.google.com/file/d/1Qcu1yHZPfU4qKlm0eIYfkjsAxfG87Rah/view?usp=sharing),
* Extract and put them in the same folder with the '*.py*' file,
* Run ```RmbBanknotesRecognition.py``` (used library: Tensoflow, OpenCV, Numpy)
<br>
Successful execution will give a result as shown below. Note that Figures (c) and (d) are intentionally removed and added one character, respectively, from Figure (b).

![](https://github.com/ardianumam/BanknotesRecognition/blob/master/Prediction_output_of_input_samples.JPG) 


## Cite the Paper Here
```
@inproceedings{umam2018BPN,
  title={A Light Deep Learning Based Method for Bank Serial Number Recognition},
  author={Umam, Ardian and Chuang, Jen-Hui and Li, Dong-Lin},
  booktitle={IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  year={2018}
}
```
The paper can be accessed [here](https://ieeexplore.ieee.org/Xplore/home.jsp) (will be updated after conference day)
