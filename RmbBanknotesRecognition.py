from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import cv2
import tensorflow as tf
FLAGS = None
import os

#global variable
cwd = os.getcwd() #current path directory
modeldir_localization = cwd + '/trained_model/localization'
modeldir_classifier = cwd + '/trained_model/char_classification'
dir_imgInput_path = 'input_img'
dir_imgOutput_path = 'output_img'
show_result = True #True for showing the result, False for not showing
potRoiCoordinate = (22, 255, 310, 359) #potentialRoI coordinate
blockW = 16; blockH = 8 #block size

#create folder for output prediction
if not os.path.exists(dir_imgOutput_path):
    os.makedirs(dir_imgOutput_path)

#main function
def main():
  #read data image
  global dir_imgInput_path, show_result
  cropMoneyAppended, potRoiOriAppended = readRmbBill(dir_imgInput_path)

  #call BPN graph (bank SN localizer)
  bankSN_ROI, potRoiPred, bill_with_bbox, bbox_pred = \
      BPN_graph(potRoiOriAppended,cropMoneyAppended)

  #call character classifier
  bankSN_pred = classifier_graph(bankSN_ROI)

  #write bank SN prediction into the bill
  for i in range(bill_with_bbox.shape[0]):
      font = cv2.FONT_HERSHEY_SIMPLEX
      bill_with_bbox_i = np.reshape(bill_with_bbox[i],(bill_with_bbox.shape[1],-1,3)).astype(np.uint8)
      cv2.putText(bill_with_bbox_i,"pred:"+bankSN_pred[i],
                  (int(bbox_pred[i,0]),int(bbox_pred[i,1])-10),
                  font, 1, (0, 0, 255), 2, cv2.LINE_AA)
      if (show_result):
        cv2.imshow("bill with bbox-"+str(i)+":", bill_with_bbox_i)
      cv2.imwrite(dir_imgOutput_path+"/out-"+str(i+1)+".jpg", bill_with_bbox_i)
  cv2.waitKey(0)


#***START*** of all localization needed functions
def readRmbBill(folder_path):
    """
    read two sides RMB bill images as the input
    :param folder_path: folder path to the RMB bill images
    :return:
        cropMoneyAppended = bill image without background, appended from all bills
        potRoiAppended/255 = Potential RoIs, appended from all bills
    """
    list_img_name = os.listdir(folder_path); list_img_name = sorted(list_img_name)
    cropMoneyAppended = np.ndarray(shape=(0,420,1068)).astype(np.uint8)
    potRoiAppended = np.ndarray(shape=(0,104,288))
    for i in range(list_img_name.__len__()):
        bill_i = cv2.imread(folder_path+"/"+list_img_name[i],0)
        potRoI_i, cropMoney_i = cropMoney(bill_i)
        cropMoneyAppended = np.concatenate((cropMoneyAppended,cropMoney_i),axis=0)
        potRoiAppended = np.concatenate((potRoiAppended, potRoI_i), axis=0)

    return cropMoneyAppended, potRoiAppended/255

def BPN_graph(potRoiOriAppended,cropMoneyAppended):
    """
    function to localize bank SN RoI
    :param potRoiOriAppended: 4 Potential RoIs
    :param cropMoneyAppended: two sides (back-front) cropped RMB bill images
    :return:
        localizedRoI: localized RoI for bank SN
        potRoiPred: prediction of PotentialRoI
        bill_with_bbox: RMB bill with its bbox
        bbox_pred: bbox coordinate -->(left,top,right,bottom)
    """
    tf.reset_default_graph()
    # input placeholder
    inputPH_potRoI = tf.placeholder(tf.float32, [None, 104 * 288])  # input potRoI
    # Build the graph for the deep net
    blockPred_conv_potRoI = deepnnSmall(inputPH_potRoI)

    # make saver variabel to restore the trained model parameters
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(modeldir_localization))
        # testing the data data
        n = 4
        localizedRoI_stack = {}
        potRoiPred_stack = np.ndarray(shape=(0,104,288,3))
        bill_with_bbox_stack = np.ndarray(shape=(0,420,1068,3))
        bbox_pred_stack = np.ndarray(shape=(0,4))
        for i in range(int(potRoiOriAppended.shape[0]/4)):
            potRoi_i = potRoiOriAppended[i*4:(i*4)+4]
            cropMoney_i = cropMoneyAppended[i*2:(i*2)+2]
            block_pred = blockPred_conv_potRoI.eval(feed_dict={inputPH_potRoI: potRoi_i.reshape((-1, 104 * 288))})
            localizedRoI, potRoiPred, bill_with_bbox, bbox_pred = \
            postProcessingRoiPred(block_pred,n,potRoi_i,cropMoney_i)
            localizedRoI = np.expand_dims(localizedRoI, axis=0)
            potRoiPred = np.expand_dims(potRoiPred, axis=0)
            bill_with_bbox = np.expand_dims(bill_with_bbox, axis=0)
            bbox_pred = np.expand_dims(bbox_pred, axis=0)
            localizedRoI_stack[str(i)]=localizedRoI
            potRoiPred_stack = np.concatenate((potRoiPred_stack, potRoiPred), axis=0)
            bill_with_bbox_stack = np.concatenate((bill_with_bbox_stack, bill_with_bbox), axis=0)
            bbox_pred_stack = np.concatenate((bbox_pred_stack, bbox_pred), axis=0)
    return localizedRoI_stack, potRoiPred_stack, bill_with_bbox_stack, bbox_pred_stack

def putOverlay(inputImage, overlayPredIn, alpha):
    """
    put red-transparent overlay into PotentialRoI, and draw bbox pred.
    :param inputImage: PotentialRoI image as input
    :param overlayPredIn: pixelwise prediction roi label with size of 104x288
    :param alpha: transparent level
    :return: original PotentialRoI image with pred overlay layer
    """
    #draw the grid
    delta_x = 16; delta_y = 8
    inputImage[::delta_y,:] = (0,0,0)
    inputImage[:,::delta_x] = (0,0,0)
    result = np.copy(inputImage)
    overlayPred = np.copy(inputImage)
    overlayPred[overlayPredIn > 0] = (0,0,255)
    cv2.addWeighted(overlayPred, alpha, result, 1 - alpha,
                    0, result)
    return result

def conv(inputBlock):
    """
    convert ROI pixelwise label with the size of 104x288 pixels
    to block ROI class label with the size of 13x18. So, each block
    size is 8x16 (8 for vertical axis).
    """
    global potRoiCoordinate, blockH, blockW
    height = potRoiCoordinate[3]-potRoiCoordinate[1]
    width = potRoiCoordinate[2]-potRoiCoordinate[0]
    result = np.full((int(height/blockH), int(width/blockW)), 0)
    for i in range(int(height/blockH)):#ver loop
        for j in range(int(width/blockW)):#hor loop
            verStart = int(i*blockH); horStart = int(j*blockW)
            result[i,j]=np.round(np.mean(
                inputBlock[verStart:verStart+blockH, horStart:horStart+blockW]))
    return result

def cropMoney(inputImage):
    """
    main function for RoI pre-processing
    :param inputImage: RMB bill image
    :return: resultImage: 2 PotentialRoIs from one side RMB bill
    """
    #find horizontal cutting point
    inputImage2 = np.copy(inputImage)#copy to second var to keep the originality
    inputImage2[inputImage2 < 80]=0
    verProjectedMoney = cv2.reduce(inputImage2, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    normalized = np.copy(verProjectedMoney)
    cv2.normalize(verProjectedMoney, normalized, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    diff = abs(np.diff(normalized.astype(np.int32), 1, axis=1)).flatten()
    loc = np.argwhere(diff >4)
    minHor = np.asscalar(np.min(loc))
    maxHor = np.asscalar(np.max(loc))

    #find vertical cutting point
    horProjectedMoney = cv2.reduce(inputImage2, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    normalized = np.copy(horProjectedMoney)
    cv2.normalize(horProjectedMoney, normalized, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    diff = abs(np.diff(normalized.astype(np.int32), 0, axis=1)).flatten()
    loc = np.argwhere(diff > 50)
    minVer = np.asscalar(np.min(loc))
    maxVer = np.asscalar(np.max(loc))
    croppedImage = inputImage[minVer:maxVer, minHor:maxHor]

    widthScalledMoney = 1068; heightScalledMoney = 420
    croppedImage = cv2.resize(croppedImage, (widthScalledMoney, heightScalledMoney))
    flipped = np.flip(np.flip(np.copy(croppedImage), axis=0), axis=1)
    flipped = flipped.astype(np.uint8)
    minHorPotROI = potRoiCoordinate[0]; minVerPotROI = potRoiCoordinate[1]
    maxHorPotROI = potRoiCoordinate[2]; maxVerPotROI = potRoiCoordinate[3]
    potROI = np.copy(croppedImage[minVerPotROI:maxVerPotROI,minHorPotROI:maxHorPotROI])
    potRoiFlipped = np.copy(flipped[minVerPotROI:maxVerPotROI,minHorPotROI:maxHorPotROI])
    potROI = np.expand_dims(potROI, axis=0)
    potRoiFlipped = np.expand_dims(potRoiFlipped, axis=0)
    PotentialRoIs = np.concatenate((potROI, potRoiFlipped), axis=0)
    croppedImage = np.expand_dims(croppedImage, axis=0).astype(np.uint8)
    return PotentialRoIs, croppedImage

def blockLabelToImage(inputBlockLabel):
    """
    convert back block RoI class label with size of 13x18 to
    original image size of PotentialRoI with size of 104x288. It will be used
    for final RoI prediction (after post-processing)
    """
    global potRoiCoordinate
    potRoiH = int(potRoiCoordinate[3]-potRoiCoordinate[1])
    potRoiW = int(potRoiCoordinate[2]-potRoiCoordinate[0])
    result = np.full((potRoiH, potRoiW), np.uint8(0))
    for i in range(inputBlockLabel.shape[0]):#ver-loop
        for j in range(inputBlockLabel.shape[1]):#hor-loop
            verStart = int(i * blockH)
            horStart = int(j * blockW)
            result[verStart:verStart+blockH,horStart:horStart+blockW]=inputBlockLabel[i,j]
    return result

def deepnnSmall(x):
  """
  BPN (Block-wise Prediction Network) to locate bank SN RoI
  :param x: PotentialRoI image
  :return: blockPred_conv for RoI with size of 13x18
  """
  with tf.name_scope('reshape'):
    do1 = 1
    x_image = tf.reshape(x, [-1, 104, 288, do1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    do2 = 10
    W_conv1 = weight_variable([5, 5, do1, do2])
    b_conv1 = bias_variable([do2])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    do3=10
    W_conv2 = weight_variable([5, 5, do2, do3])
    b_conv2 = bias_variable([do3])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('conv3'):
    do4=10
    W_conv3 = weight_variable([5, 5, do3, do4])
    b_conv3 = bias_variable([do4])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Second pooling layer.
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)

  with tf.name_scope('conv4'):
    do5=10
    W_conv4 = weight_variable([5, 5, do4, do5])
    b_conv4 = bias_variable([do5])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

  # Second pooling layer.
  with tf.name_scope('pool4'):
    h_pool4 = max_pool_1x2(h_conv4)

  with tf.name_scope('conv5'):
    do6=2
    W_conv5 = weight_variable([5, 5, do5, do6])
    b_conv5 = bias_variable([do6])
    h_conv4 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    blockPred_conv = tf.reshape(h_conv4,[-1,2])
  return blockPred_conv

def postProcessingRoiPred(predTest, n, potRoiTs, cropMoney_i):
    """
    post processing is for bounding box refinement and decide
    which Potential RoI of 4 images containing the bank serial number
    :param predTest: prediction from deepnnSmall function
    :param n: number of input images (has to be 4 --> 4 PotentialRoIs per bill)
    :return: bbox coordinate (left, top, right, bottom)
             posClass: which image contains bank serial number
    """
    predTestFlat = np.argmax(predTest, axis=1)
    predImgBlock = predTestFlat.reshape((n, 13, 18))
    lossFunctionArray = np.full((4), 0)
    predImgFinal = np.full((104,288),0); bboxFinal = np.full((4),0)
    lossFunctionTemp = 10000
    for i in range(n):
        img = predImgBlock[i, :, :]  # take block pred image one by one
        predImg = blockLabelToImage(img.reshape((13, 18)))
        kernel = np.ones((16,48), np.uint8)
        predImg = cv2.morphologyEx(predImg, cv2.MORPH_OPEN, kernel)
        connectivity = 8
        # Perform connected component to take positive pred. area
        output = cv2.connectedComponentsWithStats(predImg,
                                                  connectivity,
                                                  cv2.CV_32S)
        labels = output[1]
        stats = output[2]
        labels = labels.astype(np.uint8)
        j = 0; areaMax = 0; biggestComp = 0
        for label in np.unique(labels):
            if label == 0:
                j += 1
                continue
            area = stats[label,cv2.CC_STAT_AREA]
            if(areaMax<area):
                areaMax = area
                biggestComp = j
            j += 1
        leftConComp = stats[biggestComp, cv2.CC_STAT_LEFT]
        rightConComp = leftConComp+stats[biggestComp, cv2.CC_STAT_WIDTH]
        topConComp = stats[biggestComp, cv2.CC_STAT_TOP]
        botConComp = topConComp+stats[biggestComp, cv2.CC_STAT_HEIGHT]
        conCompWidth = rightConComp-leftConComp
        conCompHeight = botConComp-topConComp
        bbox = np.array([leftConComp, topConComp, rightConComp, botConComp])
        if (j==1):#if there is no con. component other than black pixels
            bbox = np.array([0,0,0,0])
            areaMax = 0

        #determine which one is the positive class
        lossFunction1 = abs((13*16*4*8)-areaMax)#from area
        lossFunction2=0;lossFunction3=0;lossFunction4=0
        if (conCompWidth<(11*16)):
            lossFunction2 = 1800 #from height
        if (conCompHeight>(5*8)):
            lossFunction3 = 1500 #from max height
        if (conCompHeight<(4*8)):
            lossFunction4 = 1500 #from min height
        lossFunction=lossFunction1+lossFunction2+lossFunction3+lossFunction4
        lossFunctionArray[i] = lossFunction
        if (lossFunction<lossFunctionTemp): #take with the smallest loss func.
            lossFunctionTemp = lossFunction
            predImgFinal = predImg
            bboxFinal = bbox
    posClass = np.argmin(lossFunctionArray)

#post-processing bbox prediction
  #horizontal post-processing
    #left checking
    intensityThresh = 110.0
    horThresh=(intensityThresh/255.0)
    textPxAreaThreshInSmall=10;textPxAreaThreshInFull=7;textPxAreaThreshOut = 15
    additionalHorAreaBig = 16; additionalHorAreaSmall = 8
    rangeCheckHorOut = 16; rangeCheckHorInSmall=8;rangeCheckHorInFull=16
    #checking in inside SMALL area
    checkedAreaIn = np.copy(potRoiTs[posClass,bboxFinal[1]:bboxFinal[3],
                         bboxFinal[0]:(bboxFinal[0]+rangeCheckHorInSmall)])
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn<horThresh]=1
    checkedAreaIn2[checkedAreaIn>=horThresh]=0
    potentialTextPxAreaInSmall = np.sum(checkedAreaIn2)
    #checking in inside FULL area
    checkedAreaIn = potRoiTs[posClass, bboxFinal[1]:bboxFinal[3],
                         bboxFinal[0]:(bboxFinal[0] + rangeCheckHorInFull)]
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn < horThresh] = 1
    checkedAreaIn2[checkedAreaIn >= horThresh] = 0
    potentialTextPxAreaInFull = np.sum(checkedAreaIn2)
    if (bboxFinal[0] > 0):
        # checking in outside area
        checkedAreaOut = np.copy(potRoiTs[posClass, bboxFinal[1]:bboxFinal[3],
                         (bboxFinal[0] - rangeCheckHorOut):bboxFinal[0]])
        checkedAreaOut2 = np.copy(checkedAreaOut)
        checkedAreaOut2[checkedAreaOut < horThresh] = 1
        checkedAreaOut2[checkedAreaOut >= horThresh] = 0
        potentialTextPxAreaOut = np.sum(checkedAreaOut2)
        #check cond-1 hor
        if(potentialTextPxAreaOut>textPxAreaThreshOut and potentialTextPxAreaInFull>textPxAreaThreshInFull):
            bboxFinal[0]-=additionalHorAreaBig
        #check cond-2 hor
        elif(potentialTextPxAreaInSmall>textPxAreaThreshInSmall and potentialTextPxAreaOut<=textPxAreaThreshOut):
            bboxFinal[0] -= additionalHorAreaSmall
    #check cond-3 hor
    if(potentialTextPxAreaInFull<textPxAreaThreshInFull):
        bboxFinal[0] += 12

    #right checking
    # checking in inside SMALL area
    checkedAreaIn = np.copy(potRoiTs[posClass, bboxFinal[1]:bboxFinal[3],
                    (bboxFinal[2]-rangeCheckHorInSmall):bboxFinal[2]])
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn<horThresh] = 1
    checkedAreaIn2[checkedAreaIn>=horThresh] = 0
    potentialTextPxAreaInSmall = np.sum(checkedAreaIn2)
    # checking in inside FULL area
    checkedAreaIn = potRoiTs[posClass, bboxFinal[1]:bboxFinal[3],
                    (bboxFinal[2] - rangeCheckHorInFull):bboxFinal[2]]
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn < horThresh] = 1
    checkedAreaIn2[checkedAreaIn >= horThresh] = 0
    potentialTextPxAreaInFull = np.sum(checkedAreaIn2)
    if (bboxFinal[0] < (18 * 16)):
        # checking in outside area
        checkedAreaOut = np.copy(potRoiTs[posClass, bboxFinal[1]:bboxFinal[3],
                         bboxFinal[2]:(bboxFinal[2] + rangeCheckHorOut)])
        checkedAreaOut2 = np.copy(checkedAreaOut)
        checkedAreaOut2[checkedAreaOut < horThresh] = 1
        checkedAreaOut2[checkedAreaOut >= horThresh] = 0
        potentialTextPxAreaOut = np.sum(checkedAreaOut2)
        # check con-1 hor
        if(potentialTextPxAreaInFull > textPxAreaThreshInFull and potentialTextPxAreaOut > textPxAreaThreshOut):
            bboxFinal[2] += additionalHorAreaBig
        # check cond-2 hor
        elif(potentialTextPxAreaInSmall > textPxAreaThreshInSmall and potentialTextPxAreaOut <= textPxAreaThreshOut):
            bboxFinal[2] += additionalHorAreaSmall
    # check cond-3 hor
    if(potentialTextPxAreaInFull<textPxAreaThreshInFull):
        bboxFinal[2] -= 12

#vertical post-processing
    # top checking
    verThresh = (intensityThresh / 255.0)
    textPxAreaThreshInSmall=10;textPxAreaThreshInFull=30;textPxAreaThreshOut = 20
    additionalVerAreaBig=8;additionalVerAreaSmall=4
    rangeCheckVerOut=4;rangeCheckVerInSmall=6;rangeCheckVerInFull = 8
    # checking in inside SMALL area
    checkedAreaIn = np.copy(potRoiTs[posClass, bboxFinal[1]:bboxFinal[1]+rangeCheckVerInSmall,
                    bboxFinal[0]:bboxFinal[2]])
    checkedAreaIn2 = np.copy(checkedAreaIn)
    predImgFinalPart = predImgFinal[bboxFinal[1]:bboxFinal[1]+rangeCheckVerInSmall,
                    bboxFinal[0]:bboxFinal[2]]
    checkedAreaIn[predImgFinalPart == 0] = 1  # not predicted area, ignore it
    checkedAreaIn2[checkedAreaIn < verThresh] = 1
    checkedAreaIn2[checkedAreaIn >= verThresh] = 0
    potentialTextPxAreaInSmall = np.sum(checkedAreaIn2)
    # checking in inside FULL area
    checkedAreaIn = potRoiTs[posClass, bboxFinal[1]:bboxFinal[1]+rangeCheckVerInFull,
                    bboxFinal[0]:bboxFinal[2]]
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn < verThresh] = 1
    checkedAreaIn2[checkedAreaIn >= verThresh] = 0
    potentialTextPxAreaInFull = np.sum(checkedAreaIn2)
    if (bboxFinal[1] > 0):
        # checking in outside area
        checkedAreaOut = np.copy(potRoiTs[posClass,bboxFinal[1]-rangeCheckVerOut:bboxFinal[1],
                        bboxFinal[0]:bboxFinal[2]])
        checkedAreaOut2 = np.copy(checkedAreaOut)
        predImgFinalPart = predImgFinal[bboxFinal[1]-rangeCheckVerOut:bboxFinal[1],
                        bboxFinal[0]:bboxFinal[2]]
        checkedAreaOut[predImgFinalPart == 0] = 1  # not predicted area, ignore it
        checkedAreaOut2[checkedAreaOut < verThresh] = 1
        checkedAreaOut2[checkedAreaOut >= verThresh] = 0
        potentialTextPxAreaOut = np.sum(checkedAreaOut2)
        # check cond-1 hor
        if (potentialTextPxAreaOut > textPxAreaThreshOut):
            bboxFinal[1] -= additionalVerAreaBig
        # check cond-2 hor
        elif(potentialTextPxAreaInSmall > textPxAreaThreshInSmall and potentialTextPxAreaOut <= textPxAreaThreshOut):
            bboxFinal[1] -= additionalVerAreaSmall
    # check cond-3 hor
    if (potentialTextPxAreaInFull < textPxAreaThreshInFull):
        bboxFinal[1] += 2

    # bottom checking
    # checking in inside SMALL area
    checkedAreaIn = np.copy(potRoiTs[posClass,bboxFinal[3]-rangeCheckVerInSmall:bboxFinal[3],
                    bboxFinal[0]:bboxFinal[2]])
    checkedAreaIn2 = np.copy(checkedAreaIn)
    predImgFinalPart = predImgFinal[bboxFinal[3] - rangeCheckVerInSmall:bboxFinal[3],
                       bboxFinal[0]:bboxFinal[2]]
    checkedAreaIn[predImgFinalPart == 0] = 1  # not predicted area, ignore it
    checkedAreaIn2[checkedAreaIn<verThresh] = 1
    checkedAreaIn2[checkedAreaIn>=verThresh] = 0
    potentialTextPxAreaInSmall = np.sum(checkedAreaIn2)
    # checking in inside FULL area
    checkedAreaIn = potRoiTs[posClass,bboxFinal[3]-rangeCheckVerInFull:bboxFinal[3],
                    bboxFinal[0]:bboxFinal[2]]
    checkedAreaIn2 = np.copy(checkedAreaIn)
    checkedAreaIn2[checkedAreaIn < verThresh] = 1
    checkedAreaIn2[checkedAreaIn >= verThresh] = 0
    potentialTextPxAreaInFull = np.sum(checkedAreaIn2)
    if (bboxFinal[3] < (13 * 8)):
        # checking in outside area
        checkedAreaOut = np.copy(potRoiTs[posClass,bboxFinal[3]:bboxFinal[3]+rangeCheckVerInSmall,
                        bboxFinal[0]:bboxFinal[2]])
        checkedAreaOut2 = np.copy(checkedAreaOut)
        predImgFinalPart = predImgFinal[bboxFinal[3]:bboxFinal[3]+rangeCheckVerInSmall,
                           bboxFinal[0]:bboxFinal[2]]
        checkedAreaOut[predImgFinalPart == 0] = 1  # not predicted area, ignore it
        checkedAreaOut2[checkedAreaOut < verThresh] = 1
        checkedAreaOut2[checkedAreaOut >= verThresh] = 0
        potentialTextPxAreaOut = np.sum(checkedAreaOut2)
        # check cond-1 hor
        if (potentialTextPxAreaInSmall > textPxAreaThreshInSmall and potentialTextPxAreaOut > textPxAreaThreshOut):
            bboxFinal[3] += additionalVerAreaBig
        # check cond-2 hor
        elif(potentialTextPxAreaInSmall > textPxAreaThreshInSmall and potentialTextPxAreaOut < textPxAreaThreshOut):
            bboxFinal[3] += additionalVerAreaSmall
    # check cond-3 hor
    if (potentialTextPxAreaInFull < textPxAreaThreshInFull):
        bboxFinal[3] -= 2

    #show the ROI pred. result
    img = (potRoiTs[posClass, :, :] * 255).reshape((104, -1)).astype(np.uint8)
    imgIn = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    potRoiPred = putOverlay(imgIn, predImgFinal, 0.3)
    cv2.rectangle(potRoiPred, (bboxFinal[0],bboxFinal[1]),(bboxFinal[2],bboxFinal[3]),255, 2)
    bboxFinal[bboxFinal<0]=0#make sure there is no neg bbox
    if(posClass<=1):
        bill_with_bbox = cropMoney_i[0].reshape((420,1068))
    else:
        bill_with_bbox = cropMoney_i[1].reshape((420, 1068))
    x1 = bboxFinal[0]+22; y1 = bboxFinal[1]+255
    x2 = bboxFinal[2]+22; y2 = bboxFinal[3]+255
    bbox_pred = np.array([x1,y1,x2,y2])
    localizedRoI = np.copy(bill_with_bbox[y1:y2,x1:x2])
    bill_with_bbox = cv2.cvtColor(bill_with_bbox, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bill_with_bbox,(x1, y1),(x2, y2), 255, 2)

    return localizedRoI, potRoiPred, bill_with_bbox, bbox_pred

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_1x2(x):
  """max_pool_1x2 downsamples a feature map by 2X for its hor axis."""
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#***END*** of all localization needed functions

#***START*** of all character classification needed functions
def classifier_graph(RoI):
    """
    function to read bank SN given localized RoI
    :param RoI: localized RoI
    :return: bank SN prediction result
    """
    tf.reset_default_graph()
    # Create the model
    inputChar_PH = tf.placeholder(tf.float32, [None, 24, 20])  # input potRoI
    # Build the graph for the deep net
    y_conv, keep_prob = deepnn_classifier(inputChar_PH)

    # make saver variabel to restore the trained model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(modeldir_classifier))
        bankSN_pred_list = []
        for i in range(len(RoI)):
            # separate character of given RoI
            tenChar_img, mask = separateChar(RoI[str(i)].astype(np.uint8))
            predOut = y_conv.eval(feed_dict={inputChar_PH: tenChar_img,
                                             keep_prob: 1})
            bankSN_pred = convertPredToChars(predOut)
            bankSN_pred_list.append(bankSN_pred)

    return bankSN_pred_list

def separateGroupedChars(inputImage):
    """
    function to separate grouped characters
    :param inputImage: grouped characters images
    :return: mask of separated character images
    """
    ret, inputImage = cv2.threshold(inputImage, 0, 255, cv2.THRESH_OTSU)
    kernelMorpOpen = np.full((2, 1), np.uint8(255))
    inputImage = cv2.morphologyEx(inputImage, cv2.MORPH_OPEN, kernelMorpOpen)
    inputImage = 255 - inputImage
    connectivity = 8
    output = cv2.connectedComponentsWithStats(inputImage,connectivity,cv2.CV_32S)
    labels = output[1]
    labels = labels.astype(np.uint8)
    stats = output[2]
    mask = np.zeros(inputImage.shape, dtype="uint8")
    charOrder = np.ndarray(shape=(0, 2))  # 0: left of con.comp,1: label of con.comp
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(inputImage.shape, dtype="uint8")
        labelMask[labels==label] = 255

        # if the connected component satisfies the given
        # criteria, add it to the mask
        leftConComp = stats[label, cv2.CC_STAT_LEFT]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if height > 13 and height < 28:
            mask = cv2.add(mask, labelMask)
            #save label order based on left-ness position
            charOrderCurrent = np.array([[leftConComp,label]])
            charOrder = np.concatenate((charOrder,charOrderCurrent), axis=0)
    return mask

def finishingCharSeparation(inputImage, imgOri):
    """
    function for finishing process of character separation
    :param inputImage: mask of separated character
    :param imgOri: original image of bank SN RoI
    :return: ten separated characters and its mask
    """
    connectivity = 8
    output = cv2.connectedComponentsWithStats(inputImage,connectivity,cv2.CV_32S)
    labels = output[1]
    labels = labels.astype(np.uint8)
    stats = output[2]
    mask = np.zeros(inputImage.shape, dtype="uint8")
    count=0
    charOrder = np.ndarray(shape=(0, 2))  # 0: left of con.comp,1: label of con.comp
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(inputImage.shape, dtype="uint8")
        labelMask[labels==label] = 255
        # if the connected component satisfies the given
        # criteria, add it to the mask
        leftConComp = stats[label, cv2.CC_STAT_LEFT]
        rightConComp = leftConComp + stats[label, cv2.CC_STAT_WIDTH]
        topConComp = stats[label, cv2.CC_STAT_TOP]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        botConComp = topConComp + height
        if height > 13 and height < 28\
                and rightConComp < inputImage.shape[1]\
                and topConComp > 1\
                and botConComp < inputImage.shape[0]-1:
            mask = cv2.add(mask, labelMask)
            #save label order based on left-ness position
            charOrderCurrent = np.array([[leftConComp,label]])
            charOrder = np.concatenate((charOrder,charOrderCurrent), axis=0)
            count += 1

    # put text above each chars
    tenChars = np.ndarray(shape=(0, 24, 20))
    charOrder = charOrder[np.argsort(charOrder[:, 0])]  # col0: left_loc, #col1:label of con.comp
    for i in range(charOrder.shape[0]):
        left = int(charOrder[i, 0])
        top = stats[int(charOrder[i, 1]), cv2.CC_STAT_TOP]
        right = left + stats[int(charOrder[i, 1]), cv2.CC_STAT_WIDTH]
        bottom = top + stats[int(charOrder[i, 1]), cv2.CC_STAT_HEIGHT]
        delta = 1  # additional space to crop the char area
        top = top - delta;bottom = bottom + delta;left = left - delta;right = right + delta
        # set to the max img size if the top,bot,left,right are outside the img
        if top < 0:
            top = 0
        if bottom > inputImage.shape[0]:
            bottom = inputImage.shape[0]
        if left < 0:
            left = 0
        if right > inputImage.shape[1]:
            right = inputImage.shape[1]
        charImg = np.copy(imgOri[top:bottom, left:right])
        charImg = charPreProcess(charImg)
        charImg = np.expand_dims(charImg, axis=0)
        tenChars = np.concatenate((tenChars, charImg), axis=0)
    return tenChars, mask

def charSeparationStage2(inputImage, imgOri):
    """
    function to perform second stage character separation, which is in case
    there are several characters are grouped into one connected component
    :param inputImage: binary thresholded image of bank SN RoI
    :param imgOri: original image of bank SN RoI
    :return: ten separated characters and its mask
    """
    inputImage = 255 - inputImage
    connectivity = 8
    output = cv2.connectedComponentsWithStats(inputImage,connectivity,cv2.CV_32S)
    labels = output[1]
    labels = labels.astype(np.uint8)
    stats = output[2]
    mask = np.zeros(inputImage.shape, dtype="uint8")
    count=0
    charOrder = np.ndarray(shape=(0, 2))  # 0: left of con.comp,1: label of con.comp
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(inputImage.shape, dtype="uint8")
        labelMask[labels==label] = 255

        # if the connected component satisfies the given
        # criteria, add it to the mask
        leftConComp = stats[label, cv2.CC_STAT_LEFT]
        rightConComp = leftConComp + stats[label, cv2.CC_STAT_WIDTH]
        topConComp = stats[label, cv2.CC_STAT_TOP]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        botConComp = topConComp + height
        if height > 13 and height < 48:
            # widht > 25 means several chars are grouped into 1,
            #thus, seprate it
            if width > 25:
                img = np.copy(imgOri[topConComp:botConComp,leftConComp:rightConComp])
                out = separateGroupedChars(img)
                mask[topConComp:botConComp,leftConComp:rightConComp]=out
            else:
                mask = cv2.add(mask, labelMask)
            #save label order based on left-ness position
            charOrderCurrent = np.array([[leftConComp,label]])
            charOrder = np.concatenate((charOrder,charOrderCurrent), axis=0)
            count += 1
    #perform local otsu thresholding to each character
    # and take the foreground (character itself)
    tenChars, mask = finishingCharSeparation(mask, imgOri)
    return tenChars, mask

def charSeparationStage1(inputImage, imgOri):
    """
    first stage of character separation
    :param inputImage: binary thresholded image of bank SN RoI
    :param imgOri: original image of bank SN RoI
    :return: ten separated characters and its mask
    """
    inputImage2 = np.copy(255 - inputImage)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(inputImage2, connectivity, cv2.CV_32S)
    labels = output[1]
    labels = labels.astype(np.uint8)
    stats = output[2]
    mask = np.zeros(inputImage2.shape, dtype="uint8")
    count=0
    charOrder = np.ndarray(shape=(0, 2))  # 0: left of con.comp,1: label of con.comp
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(inputImage2.shape, dtype="uint8")
        labelMask[labels==label] = 255
        # if the connected component satisfies the given
        # criteria, add it to the mask
        leftConComp = stats[label, cv2.CC_STAT_LEFT]
        rightConComp = leftConComp + stats[label, cv2.CC_STAT_WIDTH]
        topConComp = stats[label, cv2.CC_STAT_TOP]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        botConComp = topConComp + height
        if height > 13 and height < 28\
                and rightConComp < inputImage2.shape[1]\
                and topConComp > 1\
                and botConComp < inputImage2.shape[0]-1:
            mask = cv2.add(mask, labelMask)
            #save label order based on left-ness position
            charOrderCurrent = np.array([[leftConComp,label]])
            charOrder = np.concatenate((charOrder,charOrderCurrent), axis=0)
            count += 1
    if count != 10:#if several chars are grouped, perform stage-2 separation
        tenChars, mask = charSeparationStage2(inputImage, imgOri)
    else:
        tenChars = np.ndarray(shape=(0, 24, 20))
        charOrder = charOrder[np.argsort(charOrder[:, 0])]  # col0: left_loc, #col1:label of con.comp
        for i in range(charOrder.shape[0]):
            left = int(charOrder[i, 0])
            top = stats[int(charOrder[i, 1]), cv2.CC_STAT_TOP]
            right = left + stats[int(charOrder[i, 1]), cv2.CC_STAT_WIDTH]
            bottom = top + stats[int(charOrder[i, 1]), cv2.CC_STAT_HEIGHT]
            delta = 1  # additional space to crop the char area
            top=top-delta;bottom=bottom+delta;left=left-delta;right=right+delta
            # set to the max img size if the top,bot,left,right are outside the img
            if top < 0:
                top = 0
            if bottom > inputImage.shape[0]:
                bottom = inputImage.shape[0]
            if left < 0:
                left = 0
            if right > inputImage.shape[1]:
                right = inputImage.shape[1]
            charImg = np.copy(imgOri[top:bottom, left:right])
            charImg = charPreProcess(charImg)
            charImg = np.expand_dims(charImg, axis=0)
            tenChars = np.concatenate((tenChars, charImg), axis=0)
    return tenChars, mask

def charPreProcess(inputImage):
    """
    pre-process each character.
    *For char "I" or "1", apply white padding in the horizontal direction
    *For other chars, resize into fixed size of 20x24
    :param inputImage: one-separated-character image
    :return: pre-processed one-separated-character
    """
    ret, inputImage2 = cv2.threshold(inputImage, 0, 255, cv2.THRESH_OTSU)
    inputImage2 = 255 - inputImage2
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(inputImage2,
                                              connectivity,
                                              cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    labels = labels.astype(np.uint8)
    j = 0;areaMax = 0;biggestComp = 0
    for label in np.unique(labels):
        if label == 0:
            j += 1
            continue
        area = stats[label, cv2.CC_STAT_AREA]
        if (areaMax < area):
            areaMax = area
            biggestComp = j
        j += 1
    leftConComp = stats[biggestComp, cv2.CC_STAT_LEFT]
    rightConComp = leftConComp + stats[biggestComp, cv2.CC_STAT_WIDTH]
    topConComp = stats[biggestComp, cv2.CC_STAT_TOP]
    botConComp = topConComp + stats[biggestComp, cv2.CC_STAT_HEIGHT]
    if (j == 1):  # if there is no con. component other than black pixels
        bbox = np.array([0, 0, 0, 0])
        areaMax = 0
    charArea = np.copy(inputImage[topConComp:botConComp,leftConComp:rightConComp])
    ret, charAreaOtsu = cv2.threshold(charArea, 0, 255, cv2.THRESH_OTSU)
    if(stats[biggestComp, cv2.CC_STAT_WIDTH]<11):#if the char is "I", do zero padding
        charH = 24;ratioScalling = charAreaOtsu.shape[0] / charH
        charW = int((1 / ratioScalling) * charAreaOtsu.shape[1])
        result = cv2.resize(charAreaOtsu, (charW, charH))
        #left-right white padding
        leftPad = int((20-charW)/2)
        rightPad = (20-charW)-leftPad
        result = np.concatenate((np.full((24, leftPad),np.uint8(255)),result), axis=1)
        result = np.concatenate((result, np.full((24, rightPad),np.uint8(255))), axis=1)
    else:#for other than "I", do rescalling
        result = cv2.resize(charAreaOtsu, (20, 24))
    return result

listFileNames=0#only to declare as global var

def separateChar(img):
    """
    given localized RoI of bank SN, separate in chars inside it
    :param img: bank SN RoI
    :return: ten separated chars and the mask
    """
    img = np.reshape(img,(img.shape[1],-1))
    ret, img2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    kernelMorpOpen = np.full((2, 1), np.uint8(255))
    img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernelMorpOpen)
    tenChars, mask = charSeparationStage1(img2, img)
    mask = cv2.resize(mask, (232,36))
    tenChars = tenChars/255
    return tenChars, mask

dictionary = {'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7',
              '8':'8','9':'9','10':'A','11':'B','12':'C','13':'D','14':'E',
              '15':'F','16':'G','17':'H','18':'J','19':'K','20':'L','21':'M',
              '22':'N','23':'P','24':'Q','25':'R','26':'S','27':'T','28':'U',
              '29':'V','30':'W','31':'X','32':'Y','33':'Z','34':'O','35':'I'}

def convertPredToChars(pred):
    """
    convert prediction output of CNN classifier to character,
    since ouput prediction is still in numbering class
    :param pred: output prediction of CNN classifier
    :return:
    """
    predSymbol = np.argmax(pred,axis=1)
    tenChars = ''
    # check how many letters in the bank SN prediction
    # if it already has two letters, just return tenChars var
    # if < 2, check four first chars, and consider char 0 or 1 to be O or I
    n_letters = (predSymbol >= 10).sum()
    if (n_letters < 2):
        four_first = np.copy(predSymbol[0:4])
        zeroOrOne_indices = np.where(four_first<=1)[0]#return index of 0 or 1 value
        for i in range(0,(2-n_letters)):
            if(predSymbol[int(zeroOrOne_indices[i])]==0):
                predSymbol[int(zeroOrOne_indices[i])]=34
            if(predSymbol[int(zeroOrOne_indices[i])]==1):
                predSymbol[int(zeroOrOne_indices[i])]=35
    #convert to char using defined dictionary
    for i in range(predSymbol.shape[0]):
        tenChars=tenChars+dictionary[str(predSymbol[i])]
    return tenChars

def deepnn_classifier(x):
  x_image = tf.reshape(x, [-1, 24, 20, 1])
  depth1 = 32
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, depth1])
  b_conv1 = bias_variable([depth1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  depth2 = 32
  W_conv2 = weight_variable([5, 5, depth1, depth2])
  b_conv2 = bias_variable([depth2])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([5 * 6 * depth2, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 5*6*depth2])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 34])
  b_fc2 = bias_variable([34])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

#***END*** of all character classification needed functions

main()