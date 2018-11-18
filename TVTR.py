import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from functools import reduce
#from sklearn.feature_extraction import image

images = np.load('train_images.npy',  encoding='latin1')
labels = np.array(pd.read_csv('train_labels.csv', index_col=None, usecols=['Id']))
testImages = np.load('test_images.npy', encoding = 'latin1')

#image1 = (images[23][1]).reshape(100,100)
#plt.imshow(image1)


flatRawIm = []

image1 = (images[23][1]).reshape(100,100)
image1
for i in range(len(images)):
 
    flatRawIm.append((images[i][1]).reshape(100,100))
flatRawIm = np.array(flatRawIm)

#Method below was an attempt to pre-process
#patches = image.extract_patches_2d((images[23][1]).reshape(100,100), (1,1))
#img = image.reconstruct_from_patches_2d(patches, (100,100))
#plt.imshow(img)
#plt.imshow((images[23][1]).reshape(100,100))


(trainData, testData, trainLabels, testLabels) = train_test_split(flatRawIm,
labels, test_size=0.25, random_state=42)

kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier


# train the k-Nearest Neighbor classifier with the current value of `k`
#yields an error of bad dimension mismatch
model = KNeighborsClassifier(n_neighbors=4)
model.fit(trainData, trainLabels)
# evaluate the model and update the accuracies list#
#score = model.score(valData, valLabels)
#print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#accuracies.append(score)
