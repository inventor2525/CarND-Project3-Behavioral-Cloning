import numpy as np
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd
import os.path
import re

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout, Cropping2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU

#Define our training data location:
def LoadData(trainPath):
	trainCSVPath = os.path.join(trainPath,'driving_log.csv')
	trainImagesFolder = os.path.join(trainPath,'IMG')

	##################
	#Read in the data:
	##################
	print("Reading CSV...")
	trainFrame = pd.read_csv(trainCSVPath, header=None, names=['centerPath', 'leftPath', 'rightPath', 'steeringAngle', 'throttle', 'brake', 'speed'])

	#Extract file names from absolute paths, and apply new
	#root path so I can more easilly track multiple data sets:
	print("Converting Image Paths...")
	trainFrame['centerPath'] = trainFrame['centerPath'].str.replace("(.*?)(\\w*_\\d\\d\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d\\d\\.jpg)","{}{}\\2".format(trainImagesFolder,os.sep.replace("\\","\\\\")))
	trainFrame['leftPath'] = trainFrame['leftPath'].str.replace("(.*?)(\\w*_\\d\\d\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d\\d\\.jpg)","{}{}\\2".format(trainImagesFolder,os.sep.replace("\\","\\\\")))
	trainFrame['rightPath'] = trainFrame['rightPath'].str.replace("(.*?)(\\w*_\\d\\d\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d_\\d\\d\\d\\.jpg)","{}{}\\2".format(trainImagesFolder,os.sep.replace("\\","\\\\")))

	#Filter out any rows with missing images
	#in-case of manual data deletions or copying:
	print("Removing Bad Rows...")
	rowsToRemove = []
	for index, row in trainFrame.iterrows():
		if not os.path.exists(row['centerPath']) or not os.path.exists(row['leftPath']) or not os.path.exists(row['rightPath']):
			rowsToRemove.append(index)
	trainFrame.drop(rowsToRemove)

	#Load data into system memory (I don't normally see this done, but I
	#have a lot of system ram, and it beats repeatedly loading from HD, though
	#probably overkill since I'm using a generator now):
	print("Loading All To RAM...")
	allData = []
	for index, row in trainFrame.iterrows():
		allData.append({"centerImage" : imread(row['centerPath']),
					  "leftImage" : imread(row['leftPath']),
					  "rightImage" : imread(row['rightPath']),
					  "steeringAngle" : row['steeringAngle']})
	if len(allData) is 0:
		print("Error: No Data Found!")
	else:
		print("Loaded!")
	return allData

#Provide means for Keras to load augmented
#batches into GPU memory as needed:
def BatchGenerator(data, batchSize=16):
	count = len(data)
	while True:
		shuffle(data)
		for batchIndex in range(0, count-batchSize-1, batchSize):
			trainX = []
			trainY = []
			for i in range(0,batchSize):
				row = data[batchIndex+i]

				trainX.append(row["centerImage"])
				trainY.append(row["steeringAngle"])

				trainX.append(row["leftImage"])
				trainY.append(row["steeringAngle"]+0.2)

				trainX.append(row["rightImage"])
				trainY.append(row["steeringAngle"]-0.2)
				
				trainX.append(np.fliplr(row["leftImage"]))
				trainY.append(-row["steeringAngle"]-0.2)

				trainX.append(np.fliplr(row["rightImage"]))
				trainY.append(-row["steeringAngle"]+0.2)

				trainX.append(np.fliplr(row["centerImage"]))
				trainY.append(-row["steeringAngle"])
			trainX = np.array(trainX)
			trainY = np.array(trainY)
			yield (trainX,trainY)

def ModelData(allData):
	print("Training...")

	#Split our data:
	print("Splitting Data...")
	shuffle(allData)
	imageSize = allData[0]["centerImage"].shape
	trainData, validationData = train_test_split(allData, test_size=0.2)

	##################
	#Define our model:
	##################
	model = Sequential()
	
	#model started as NVidia's End To End model, then
	#shunk a bit to fit in memory more easilly. (works well if given lots of data)
	#model.add(Lambda(lambda x: x/255.0-0.5, input_shape=imageSize))
	model.add(BatchNormalization(input_shape=imageSize, axis=1))
	model.add(Cropping2D(cropping=((50,24),(0,0))))
	model.add(Conv2D(6,1,1, border_mode="same"))
	model.add(LeakyReLU())
	model.add(Conv2D(12,3,3, border_mode="same"))
	model.add(LeakyReLU())
	model.add(Conv2D(24,3,3, border_mode="same"))
	model.add(LeakyReLU())
	#model.add(Conv2D(64,3,3, border_mode="same"))
	#model.add(LeakyReLU())
	#model.add(Conv2D(64,3,3, border_mode="same"))
	#model.add(LeakyReLU())

	model.add(Flatten())

	#model.add(Dense(1164))
	model.add(Dropout(0.2))
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Dense(1))
	
	#First model attempt
	#model.add(Conv2D(6,5,5))
	#model.add(LeakyReLU())
	#model.add(Conv2D(16,5,5))
	#model.add(LeakyReLU())
	#
	#model.add(Flatten())
	#model.add(Dense(128))
	#model.add(Dropout(0.3))
	#model.add(Dense(1))

	##################
	#Train our model:
	##################
	print("Fitting Model...")
	model.compile(optimizer='adam', loss='mse')
	#model.fit(trainX, trainY, validation_split=0.2, shuffle=True, nb_epoch=2)
	augmentationFactor = 6
	model.fit_generator(BatchGenerator(trainData), validation_data=BatchGenerator(trainData),
						samples_per_epoch=len(trainData)*augmentationFactor, nb_val_samples=len(validationData)*augmentationFactor,
						nb_epoch=20)
	return model
def SaveModel(model):
	try:
		print("Saving Model...")
		model.save("model.h5")
		print("Done!")
	except:
		print("Failed to save model!")
	return model;
#ModelData(LoadData())
#SaveModel(ModelData(LoadData(r'C:\Users\Charlie\Desktop\Training')))