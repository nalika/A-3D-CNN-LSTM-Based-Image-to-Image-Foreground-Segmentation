
# -*- coding: utf-8 -*-
"""
Akilan, Thangarajah @2018

This is a temporary script file.
"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv3D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import ConvLSTM2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import Conv3DTranspose
from tensorflow.python.keras.layers import BatchNormalization



class Models:
	

	def sendec_block(input_tensor1, input_tensor2):		 	
		x = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), strides=(1, 2, 2), 
			padding='same', data_format='channels_last')(input_tensor1) 
		x = concatenate([input_tensor2, x], axis=-1) 
		x = BatchNormalization()(x)	
		x = Conv3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), 
			activation='relu', padding='same', data_format='channels_last')(x)

		return x



	def _sEnDec_cnn_lstm(input_dim, dp):

		print('[INFO] Creating sEnDec_cnn_lstm Model...\n')
		input_layer = Input(shape=input_dim)	
		seq0 = Conv3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1),
		               activation='relu',
		               padding='same', data_format='channels_last')(input_layer)		

		seq1 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq0)  	

		# - SEnDec block 1
		seq12 = Models.sendec_block(seq1, input_layer)


		seq13 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq12)  
		seq2 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq13) 
		
		# - SEnDec block 2
		seq22 = Models.sendec_block(seq2, seq13)
				

		seq22 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq22)  
		seq30 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq22) 
		
		# - SEnDec block 3
		seq32 = Models.sendec_block(seq30, seq22)


		seq3 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2),
		               activation='relu',
		               padding='same', data_format='channels_last')(seq32) 
		seq4 = ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(2, 2),
		        activation='relu', padding='same', return_sequences=True)(seq3) 
				
		#-~~~~~~~~~~~~~~~~~~ Upsampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		      	               
		seq6 = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), 
				strides=(1, 2, 2), padding='same', data_format='channels_last')(seq4) 	
		seq6 = concatenate([seq6, seq3], axis=-1) 
		
		seq6 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last')(seq6)
		seq6 = BatchNormalization()(seq6)
		seq6 = Activation('relu')(seq6)
		seq6 = concatenate([seq6, seq30], axis=-1)       
		
		seq7 = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), 
				strides=(1, 2, 2), padding='same', data_format='channels_last')(seq6) 
		seq7 = concatenate([seq7, seq22], axis=-1) 
		              
		seq7 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last')(seq7)
		seq7 = BatchNormalization()(seq7)
		seq7 = Activation('relu')(seq7)
		seq7 = concatenate([seq7, seq2], axis=-1) 
		
		seq8 = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), 
				strides=(1, 2, 2), padding='same', data_format='channels_last')(seq7)  
		seq8 = concatenate([seq8, seq13], axis=-1) 
		seq8 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last')(seq8)
		
		seq8 = BatchNormalization()(seq8)
		seq8 = Activation('relu')(seq8)
		seq8 = concatenate([seq8, seq1], axis=-1) 
		
		seq9 = Conv3DTranspose(filters=16, kernel_size=(2, 3, 3), 
				strides=(1, 2, 2), padding='same', data_format='channels_last')(seq8) 
		seq9 = concatenate([seq9, seq0], axis=-1) 
		seq9 = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last')(seq9)
		
		seq9 = BatchNormalization()(seq9)
		seq9 = Activation('relu')(seq9)
		

		seq91 = Dropout(dp)(seq9)

		output_layer = Conv3D(filters=1, kernel_size=(2, 3, 3), strides=(1, 1, 1),
		               activation='sigmoid',
		               padding='same', data_format='channels_last')(seq91) #240 x 320

	

		print('[INFO] Model Creation is Completed\n')

		return Model(input_layer, output_layer)
