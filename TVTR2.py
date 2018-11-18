import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
#from sklearn import preprocessing

trainImg = np.load('train_images.npy',  encoding='latin1')
trainLabels = np.array(pd.read_csv('train_labels.csv', usecols=['Category']))
testImg = np.load('test_images.npy',  encoding='latin1')

#image1 = (images[23][1]).reshape(100,100)
#plt.imshow(image1)

#saving the resized array of every image in an array
rawIm = []
testIm = []
for i in range(10000):
    rawIm.append(trainImg[i][1])
    testIm.append(testImg[i][1])
rawIm = np.array(rawIm)
testIm = np.array(testIm)


#Testing with Linear SVC
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(rawIm, trainLabels)

print(clf.predict(testIm))

#Testing with kNN


model = KNeighborsClassifier(n_neighbors=4)
model.fit(trainData, trainLabels)
          