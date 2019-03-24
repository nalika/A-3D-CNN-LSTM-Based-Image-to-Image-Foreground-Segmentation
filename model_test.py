# -*- coding: utf-8 -*-
"""
Akilan, Thangarajah @2018

This is a temporary script file for T. Akilan, Q. J. Wu, A. Safaei, J. Huo and Y. Yang, "A 3D CNN-LSTM-Based Image-to-Image Foreground Segmentation," in IEEE Transactions on Intelligent Transportation Systems.
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8671459&isnumber=4358928
"""


# Import necessary libraries
import os
import sys
import numpy as np 
import copy

import tensorflow as tf

from tensorflow.python.keras.utils import multi_gpu_model
import argparse

from models_clean import Models

#---- TO DO -----
# Import more required libraries



## ------------ Inernal Configurations ----------------------------
IMG_HEIGHT, IMG_WIDTH, C = 240, 320, 3
n_look_back = 4
dp = 0.7
	


if __name__ == '__main__':
	
	print('\n\n\n[PROG] Excecution of main()...\n')	
	
	## -----------------End of Internal Configurations --------------------#
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-g", "--gpus", type=int, default=0,
		help="GPU ID")	
	ap.add_argument("-ms", "--model_summary", required=False, default=False,
		help="for model summary set -ms True")	
	args = vars(ap.parse_args())
	
	# grab the external arguments and store them in a conveience variables
	G = args["gpus"]	
	MODEL_SUMMARY = args["model_summary"]
	
	# Set the environment
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
	os.environ["CUDA_VISIBLE_DEVICES"] = str(G)
	

	#- Creating trainable model	
	input_dim = [n_look_back, IMG_HEIGHT, IMG_WIDTH, C]
	model = Models._sEnDec_cnn_lstm(input_dim, dp)

	###- Check the model structure	
	if MODEL_SUMMARY:
		model.summary()
		
	exit()

	#--- TO DO ----
	# Write your rest of the methods for traning and testing here after

	
