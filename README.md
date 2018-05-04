# Cloth-Classification-using-LSTM-GRU
Cloth Classification using Deep Recurrent Neural Networks and Tensorflow.

Dataset : https://goo.gl/dYFTP7

train_images: 60000 examples of 28x28 grayscale image. The data is of size 60000 x 784. Each
              pixel has a single intensity value which is an integer between 0 and 255.  
	      
train_labels: 60000 labels from 10 classes for the images in given in train_images. Class details
              are mentioned below.              
	      
test_images:  10000 examples of 28x28 grayscale image. The data is of size 10000 x 784. Each
              pixel has a single intensity value which is an integer between 0 and 255.
	      
test_labels:  10000 labels from 10 classes for the images in given in test_images. Class details
              are mentioned below.

Labels: Each training and test example is assigned to one of the following labels:

        0 T-shirt/top	
        1 Trouser
        2 Pullover
        3 Dress
        4 Coat
        5 Sandal	
        6 Shirt	
        7 Sneaker	
        8 Bag	
        9 Ankle boot

dataloader.py : Function for reading data from zip files as well as for creating mini-batches. It will work if all the zip files for train and test data are in the data directory.

main.py : A 28*28 image is treated as a sequence of 28 vectors of dimension 28. Create two separate recurrent neural networks with : 
	
	1. Standard LSTM Cell	
	2. Standard GRU Cell
	
The cells are written from scratch using tf.contrib.rnn.static_rnn instead of f.contrib.rnn.LSTMCell or tf.contrib.rnn.GRUCell to build a recurrent neural network using the specific cell.

How to run : 

	python main.py --train --hidden_unit 128 --model lstm
	python main.py --train --hidden_unit 128 --model lstm
	python main.py --test --hidden_unit 128 --model gru
	python main.py --test --hidden_unit 128 --model gru	     

Plot.py : Plot test_accuracy vs hidden_unit_size and compare
