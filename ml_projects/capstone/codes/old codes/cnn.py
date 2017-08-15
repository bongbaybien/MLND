import tensorflow as tf
import pickle
import support_functions as sf
import random

def conv_net(x, keep_prob):
    '''
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    '''
    conv_maxpool1 = conv2d_maxpool(x, conv_num_outputs=20, conv_ksize=(4,4), conv_strides=(1,1), pool_ksize=(2,2), pool_strides=(2,2))
	
    flat = flatten(conv_maxpool1)
    print('\n network shape after flat layer:', flat.get_shape)
    
    fc = fully_conn(flat, num_outputs=100)
    print('\n network shape after fc layer:', fc.get_shape)
	
    fc = tf.nn.dropout(fc, keep_prob)
    print('\n network shape after dropout layer:', fc.get_shape)
	
    out = output(fc, num_outputs=7)
    print('\n network shape after output layer:', out.get_shape)
    
    return out

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    '''
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    '''
    weight = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], tf.to_int32(x_tensor.shape[3]), conv_num_outputs]
                                             , stddev=0.1)) # (height, width, input_depth, output_depth)
    bias = tf.Variable(tf.zeros([conv_num_outputs]))
    
    conv2d = tf.nn.conv2d(x_tensor, weight, strides=[1] + list(conv_strides) + [1], padding='SAME')
    conv2d = tf.nn.bias_add(conv2d, bias)
    conv2d = tf.nn.relu(conv2d)
    print('\n network shape after conv layer:', conv2d.get_shape)
	
    maxpool = tf.nn.max_pool(conv2d, ksize=[1] + list(pool_ksize) + [1], strides=[1] + list(pool_strides) + [1], padding='SAME')
    print('\n network shape after pool layer:', maxpool.get_shape)
        
    return maxpool

def flatten(x_tensor):
    '''
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    '''
    tensor_shape = x_tensor.get_shape().as_list() 
    n_input = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]    
    x_flat = tf.reshape(x_tensor, [-1, n_input])
    return x_flat

def fully_conn(x_tensor, num_outputs):
    '''
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    '''
    tensor_shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([tensor_shape[1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros([num_outputs]))
    fc = tf.add(tf.matmul(x_tensor, weight), bias)
    fc = tf.nn.relu(fc)
    return fc

def output(x_tensor, num_outputs):
    '''
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    '''
    tensor_shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([tensor_shape[1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, weight), bias)
    return out

def test_model(data_folder, model_name, batch_size):
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open(data_folder + '/test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(model_name + '.meta')
        loader.restore(sess, model_name)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in sf.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))
        

