def make_5d(data, n_look_back):
	#~ print('Reshaping data as 5D...\n')
	#~ n_look_back = 2
	k = 0
	n_samples = len(data)
	tmp = []
	data_5d = np.empty((n_samples-(n_look_back-1), n_look_back, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
	
	for i in range(0, n_samples):
		tmp = data[i:i+n_look_back]

		if tmp.shape[0] == n_look_back:
			# print('[INFO] tmp dim : ', tmp.shape)
			for rotate_channel_id in range(0, n_look_back): # rotate the channels such that bring the current input as first channel
				# print('[]INFO] n_look_back-1-rotate_channel_id: ', n_look_back-1-rotate_channel_id)
				tmp[rotate_channel_id] = tmp[n_look_back-1-rotate_channel_id]
			tmp = tmp.reshape(1, n_look_back, data.shape[1], data.shape[2], data.shape[3])
			data_5d[k] = tmp
			tmp = [] # clear tmp
			k = k + 1
		#~ else:
			#~ tmp = np.vstack((tmp, tmp))
		#~ print(tmp.shape)
		
	#~ print('returning data with dim of :' + str(data_5d.shape) + '.\n')
	return data_5d
