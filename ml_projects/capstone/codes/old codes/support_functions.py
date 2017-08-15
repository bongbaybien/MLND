import csv
import numpy as np
from sklearn import preprocessing
import pickle

def read_in_data(file_path):	
	feature_list = []
	label_list = []
	usage_list = []

	with open(file_path) as file:
		reader = csv.reader(file)
		next(reader, None) # skip the headers
		for row in reader:			
			feature_list.append([int(p) for p in row[1].split()])
			label_list.append(int(row[0]))
			usage_list.append(row[2])
			
	return feature_list, label_list, usage_list
	
def preprocess(data, image_shape, enc_map):
    
	feature_list, label_list, usage_list = data	
	
	# remove blank images
	blank_indices = set([index for index, x in enumerate(feature_list) if max(x)==min(x)])
	feature_list = [x for index, x in enumerate(feature_list) if index not in blank_indices]
	label_list = [x for index, x in enumerate(label_list) if index not in blank_indices]	
	usage_list = [x for index, x in enumerate(usage_list) if index not in blank_indices]
	
	# reshape feature_list
	n_example = len(feature_list)
	features = np.reshape(feature_list, (n_example, *image_shape))
	
	# normalize & one hot encode
	features = normalize(features)
	labels = one_hot_encode(enc_map, label_list)

	return features, labels, usage_list

def normalize(x):
    '''
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (width, height, channel)
    : return: Numpy array of normalize data
    '''
    return np.array([(i - i.min())/(i.max() - i.min()) for i in x])

def one_hot_encode(enc_map, x):
    '''
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    '''  
    lb = _fit_lb(enc_map)
    return lb.transform(x)

def _fit_lb(enc_map):
    lb = preprocessing.LabelBinarizer()  
    lb.fit(enc_map)
    return lb

def create_train_test_data(data, folder_path):
	'''
	Training set = 'Training', validation set = 'PublicTest', test set = 'PrivateTest'.
	Note, this is not strictly the same as winning solution
	'''
	features, labels, usage_list = data
	
	train_indices = [index for index, x in enumerate(usage_list) if x=='Training']	
	train_features = features[train_indices]
	train_labels = labels[train_indices]
	pickle.dump((train_features, train_labels), open(folder_path + '/train.p', 'wb'))
	
	valid_indices = [index for index, x in enumerate(usage_list) if x=='PublicTest']	
	valid_features = features[valid_indices]
	valid_labels = labels[valid_indices]
	pickle.dump((valid_features, valid_labels), open(folder_path + '/valid.p', 'wb'))
	
	test_indices = [index for index, x in enumerate(usage_list) if x=='PrivateTest']
	test_features = features[test_indices]
	test_labels = labels[test_indices]
	pickle.dump((test_features, test_labels), open(folder_path + '/test.p', 'wb'))
	
def load_preprocess_training_batch(folder_path, batch_size):
    '''
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    '''
    filename = folder_path + '/train.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)
	
def batch_features_labels(features, labels, batch_size):
    '''
    Split features and labels into batches
    '''
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

