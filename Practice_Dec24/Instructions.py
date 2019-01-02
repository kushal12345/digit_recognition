# Train and test your classifier from a real-world data

#####################################
# Step 0: Load the necessary libraries
#####################################

import scipy.io 
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline 				# for Jupyter notebooks only

from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 


################################
# Step 1: Load the train data file
################################

# Note that the data is in .mat format
# Hint: use scipy.io.loadmat() to load the image
# import scipy.io
# train_data = ...




# Question: How many classes are there?
# Question: How many digits are there for training?

# Question: What data-structure is used to store the data and the labels?


##############################
# Step 2: Understand your data
##############################

# Hint: %who or %whos can be helpful in IPython notebook


#####################################################
# Step 3: Extract the images and labels from your data
#####################################################

# X = ...
# y = ...

#####################################################
# Step 4: View one image to make sure what the data is
#####################################################

# view an image (e.g. 25) and print its corresponding label
#img_index = 25
#plt.imshow(X[:,:,:,img_index])
#plt.show()
#print(y[img_index])


#########################################################################################
# Step 5: Reshape your matrices into 1D vectors and shuffle (still maintaining the index pairings)
#########################################################################################

#X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T 
#y = y.reshape(y.shape[0],) 
#X, y = shuffle(X, y, random_state=42)  # use a seed of 42 to replicate the results of tutorial


######################################################################################
# Step 6 (optional): Reduce dataset to a selected size (rather than all 500k examples)
######################################################################################

#size = X.shape[0] 	# change to real number to reduce size
#X = X[:size,:] 		# X.shape should be (num_examples,img_dimensions*colour_channels)
#y = y[:size] 		# y.shape should be (num_examples,)


###############################################
# Step 7: Split data into training and test set
###############################################
#
#
#
#


################################################################
#Step 8: Define your classifier and view specs of the classifier
################################################################
#
#
#
#
#################################################################
# Step 9: Fit the model on training data and predict on test data
#################################################################
#
#
#
##########################################
# Step 10: Compute accuracy of your model
##########################################