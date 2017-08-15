import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Cropping2D, Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dropout, Dense
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as k
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
import keras
import h5py
import pdb
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
seed = 1

def read_in_data(file_path):	
	x_list = []
	y_list = []
	usage_list = []

	with open(file_path) as file:
		reader = csv.reader(file)
		next(reader, None) # skip the headers
		for row in reader:			
			x_list.append([int(p) for p in row[1].split()])
			y_list.append(int(row[0]))
			usage_list.append(row[2])
			
	return x_list, y_list, usage_list
	
def format(data, image_shape, n_classes):
    
	x_list, y_list, usage_list = data	
	
	# remove blank images
	blank_indices = set([index for index, x in enumerate(x_list) if max(x)==min(x)])
	x_list = [x for index, x in enumerate(x_list) if index not in blank_indices]
	y_list = [x for index, x in enumerate(y_list) if index not in blank_indices]	
	usage_list = [x for index, x in enumerate(usage_list) if index not in blank_indices]
		
	# reshape x_list & y_list to numpy array
	n_example = len(x_list)
	x_array = np.reshape(x_list, (n_example, *image_shape))
	y_array = np.reshape(y_list, (n_example, 1))
	
	return x_array, y_array, usage_list
	
def create_train_test(data, n_classes):
	x_array, y_array, usage_list = data
	
	# split data
	train_indices = [index for index, x in enumerate(usage_list) if x=='Training']
	test_indices = [index for index, x in enumerate(usage_list) if x=='PublicTest']
	private_indices = [index for index, x in enumerate(usage_list) if x=='PrivateTest']
	
	x_train = x_array[train_indices]
	y_train = y_array[train_indices]
	
	x_test = x_array[test_indices]
	y_test = y_array[test_indices]
	
	x_private = x_array[private_indices]
	y_private = y_array[private_indices]
	
	# # number of example for each class
	# label, count = np.unique(y_train, return_counts=True)
	# print('training distribution before over sample: \n', dict(zip(label, count)))
	
	# # over sample minority class
	# n_train = len(x_train)
	# image_shape = x_train.shape[1:]
	# ros = RandomOverSampler(random_state=seed)
	# x_train, y_train = ros.fit_sample(np.reshape(x_train, (n_train, np.product(image_shape)))
									# , y_train.reshape(-1))
	
	# # number of example for each class
	# label, count = np.unique(y_train, return_counts=True)
	# print('training distribution after over sample: \n', dict(zip(label, count)))
									
	# # reshape
	# n_train = len(x_train)
	# x_train = np.reshape(x_train, (n_train, *image_shape))
	# y_train = np.reshape(y_train, (n_train, 1))
	
	return (x_train, y_train), (x_test, y_test), (x_private, y_private)

def build_model(name, kernel_initializer=TruncatedNormal(seed=1), kernel_regularizer=None, batch_norm=False):
	image_shape = (48,48,1)
	n_classes = 7
	model_name = name
	model = Sequential(name=model_name)
	model.add(Cropping2D(cropping=3, input_shape=image_shape, name='block1_crop'))
	model.add(Conv2D(32, (5,5), padding='same', kernel_initializer=kernel_initializer
				, kernel_regularizer=kernel_regularizer
				, name='block1_conv'))
	if batch_norm==True:
		model.add(BatchNormalization(axis=3, name='block1_bn'))
	model.add(LeakyReLU(name='block1_relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2), padding='valid', name='block1_pool'))

	model.add(ZeroPadding2D(padding=1, name='block2_pad'))
	model.add(Conv2D(32, (4,4), padding='valid', kernel_initializer=kernel_initializer
				, kernel_regularizer=kernel_regularizer
				, name='block2_conv'))
	if batch_norm==True:
		model.add(BatchNormalization(axis=3, name='block2_bn'))
	model.add(LeakyReLU(name='block2_relu'))
	model.add(AveragePooling2D((2,2), strides=(2,2), padding='valid', name='block2_pool'))

	model.add(Conv2D(64, (5,5), padding='same', kernel_initializer=kernel_initializer
				, kernel_regularizer=kernel_regularizer
				, name='block3_conv'))
	if batch_norm==True:
		model.add(BatchNormalization(axis=3, name='block3_bn'))
	model.add(LeakyReLU(name='block3_relu'))
	model.add(AveragePooling2D((2,2), strides=(2,2), padding='valid', name='block3_pool'))

	model.add(Flatten(name='block4_flat'))
	# TODO is this where dropout should be?
	model.add(Dropout(0.2, seed=seed, name='block4_dropout'))
	model.add(Dense(3072, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name='block4_fc'))
	# if batch_norm==True:
		# model.add(BatchNormalization())
	model.add(LeakyReLU(name='block4_relu'))
	# TODO svm Activations function
	model.add(Dense(n_classes, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name='block4_out'))
	# if batch_norm==True:
		# model.add(BatchNormalization())
	model.add(Activation('softmax', name='block4_softmax'))
    
	return model	

def center_std(x):
	return (x - x.mean())/x.std()

def preprocess(x_train, **kwargs):   
	# print('passed preprocessing_function', preprocessing_function)
	
	train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True
									, horizontal_flip=True
									, **kwargs)
	train_datagen.fit(x_train, augment=True, seed=seed)

	test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	test_datagen.fit(x_train)
    
	return train_datagen, test_datagen
    
def train(data_formatted, model, cp_name, class_weight=None
			, lr_initial=0.01, lr_monitor='val_loss', lr_factor=0.5, lr_patience=5, lr_min=1e-07
			, cp_monitor='val_loss'
			, es_monitor='val_loss', es_patience=21
			, epochs=500
			, **kwargs):
	
	n_classes = 7	
	batch_size = 256
	
	# split into train & test sets. Test set = public, private set is used for final testing, to be consistent with the competition
	(x_train, y_train), (x_test, y_test), (x_private, y_private) = create_train_test(data_formatted, n_classes)
	
	n_train = len(x_train)
	n_test = len(x_test)
	
	# one-hot encode
	y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
	y_test_onehot = keras.utils.to_categorical(y_test, n_classes)
	y_private_onehot = keras.utils.to_categorical(y_private, n_classes)

	# preprocess
	train_datagen, test_datagen = preprocess(x_train, **kwargs)
	# to use tensorboard due to this issue: https://github.com/fchollet/keras/issues/3358
	# x_test_center_std = center_std(x_test)

	# compile model
	sgd = optimizers.SGD(lr=lr_initial, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	# callbacks
	reduce_lr = ReduceLROnPlateau(monitor=lr_monitor, factor=lr_factor, patience=lr_patience, min_lr=lr_min
									, verbose=1)
	checkpointer = ModelCheckpoint(filepath=cp_name, monitor=cp_monitor, verbose=1, save_best_only=True)
	early_stopper = EarlyStopping(monitor=es_monitor, patience=es_patience)
	# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=batch_size, write_graph=True
								# , write_grads=True, write_images=True, embeddings_freq=0)
	# callbacks = [checkpointer, reduce_lr, early_stopper, tensorboard]
	callbacks = [checkpointer, reduce_lr, early_stopper]

	# fit model
	print('Start learning rate:', k.get_value(sgd.lr))
	history = model.fit_generator(train_datagen.flow(x_train, y_train_onehot, batch_size=batch_size, seed=seed)
								, steps_per_epoch=n_train//batch_size
								, epochs=epochs
								, validation_data=test_datagen.flow(x_test, y_test_onehot, batch_size=batch_size, seed=seed)
								, validation_steps=n_test//batch_size
								# , validation_data = (x_test_center_std, y_test_onehot)
								, callbacks=callbacks
								, class_weight=class_weight)
	
	print('private test metrics', model.evaluate_generator(test_datagen.flow(x_private, y_private_onehot, batch_size=batch_size, seed=seed)
                                       , steps=batch_size))
	return history
	
def visualize_metrics(history):
    # visualize loss
    plt.figure(figsize=(16, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc'], loc='upper left')

    plt.tight_layout()
