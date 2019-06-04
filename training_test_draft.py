def tain_model():
	if PRETRAIN == True:		
		weight_model_path = (<FULL_PATH_TO_THE_WEIGHTMODEL>)
		print('-'*30)
		print('[INFO] Loading pretrained weights from - ' + weight_model_path)

		model.load_weights(weight_model_path)
   	
	'''CURRENT WEIGHTs'''
	weight_model = (<FULL_PATH_TO_THE_NEW_WEIGHTMODEL>)

	if OPTIMIZER == "adam":
   		opti_algo = Adam(lr)
   		model.compile(optimizer=opti_algo, loss='binary_crossentropy', metrics=[IoU])
   		weight_saver = ModelCheckpoint(weight_model, monitor='val_IoU', save_best_only=True, save_weights_only=True)	
	else:
   		opti_algo = Adadelta(lr=lr, decay=1e-6, rho=0.9)		
   		model.compile(optimizer=opti_algo, loss=loss_fn, metrics= metrics)   		
   		weight_saver = ModelCheckpoint(weight_model, save_best_only=True, save_weights_only=True)
      
  #NOTE: Add more customized optimizers as per need
                                              
	annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
	#  If steps_per_epoch is set, the `batch_size` must be None.
	
	print('-'*30)
	print('[PROG] Model training in progress w/o data augmentation...')
	tr_time_start = time.time()
	            
	hist = model.fit(x_train,y_train, batch_size = BATCH_SIZE,
                           steps_per_epoch = None,
                           validation_data = (x_val[0:10], y_val[0:10]),
                           epochs=EPOCHS, verbose=2,
                           callbacks = [weight_saver])
	ttl_tr_time = time.time()-tr_time_start	
	print('\n[INFO] Total train time: ', ttl_tr_time)		
	
	##- Save History
	with open(<FULL_PATH_TO_SAVE_TRAINING_HISTORY>, 'wb') as file_pi:
		pickle.dump(hist.history, file_pi)
		print('\n[PROG] Saving model: ' + weight_model + '\n')
    
	with open(<FULL_PATH_TO_SAVE_TRAINING_HISTORY>, 'rb') as fp:
		hist = pickle.load(fp)
    
  ##- Setting up a plot for loss and training acc. graphs
	#--------------------------------------------------------        
	plt.plot(hist['loss'], linewidth=2, color='b', label = 'train')
	plt.plot(hist['val_loss'], linewidth=2, color='r', label = 'test')

	plt.grid()
	#~ plt.grid(linestyle='dotted')
	plt.grid(color='black', linestyle='--', linewidth=1)
	plt.ylabel('Loss', fontsize=18)
	plt.xlabel('Epoch', fontsize=18)
	plt.xticks(fontsize=12, rotation=0)
	plt.yticks(fontsize=12, rotation=0)
	plt.title(data_set_name + '_loss')
	plt.legend(shadow=False, fancybox=False) 
	plt.tight_layout()
	#~ plt.show()
	plt.savefig(<FULL_PATH_TO_SAVE_TRAINING_LOSS_PLOT>+ '.png')
	plt.close()
	
	if OPTIMIZER == "adam":
		plt.plot(hist['IoU'], linewidth=3, color='b', label = 'Train')
		plt.plot(hist['val_IoU'], linewidth=3, color='r', label = 'Valida.')
	elif loss_fn == binary_crossentropy_with_logits:
		plt.plot(hist['binary_accuracy'], linewidth=3, color='b', label = 'Train')
		plt.plot(hist['val_binary_accuracy'], linewidth=3, color='r', label = 'Valida.')
	else:
		plt.plot(hist['acc'], linewidth=3, color='b', label = 'Train')
		plt.plot(hist['val_acc'], linewidth=3, color='r', label = 'Valida.')

	plt.grid()
	plt.grid(color='black', linestyle='--', linewidth=1)
	plt.ylabel('Figure of Merit', fontsize=18)
	plt.xlabel('Epoch', fontsize=18)
	plt.xticks(fontsize=12, rotation=0)
	plt.yticks(fontsize=12, rotation=0)
	plt.title(data_set_name + '_fom')
	plt.legend(shadow=False, fancybox=False, loc='lower right')
	plt.tight_layout()
	plt.savefig(<FULL_PATH_TO_SAVE_TRAINING_FOM_PLOT> + '.png')
	#~ plt.close()


def test_model():

	line = '-'*30
	print(line)
	line = '[PROG] Loading saved weights...'
	print(line)
  
	weight_model = (<FULL_PATH_TO_LOAD_SAVED_MODEL>)
	
	line = '[INFO] weight_model: ' + weight_model + '\n'
	print(line)

	model.load_weights(weight_model)
	
	line = '-'*30+'\n'+'[PROG] Model testing in progress...'
	print(line)
	
	start_time = time.time()
	
	y_hat = model.predict(x_test, batch_size=BATCH_SIZE*2, steps=None, verbose=1)
	end_time = time.time()
	
