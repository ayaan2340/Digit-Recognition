import sklearn as sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from mnist import MNIST
import matplotlib.pyplot as plt
mndata = MNIST('data')

images, labels = mndata.load_training()

testImages, testLabels = mndata.load_testing()

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.axis('off')
    pixels = np.array(images[i])
    pixels = pixels.reshape(28,28)
    plt.imshow(pixels, cmap=plt.cm.gray_r)
    plt.title('Sample: ' + str(labels[i]))
plt.show()

clf = make_pipeline(StandardScaler(), sklearn.svm.SVC(kernel='rbf', C=1))
clf.fit(images, labels)
print(str(clf.score(testImages, testLabels)( + '% accurate')))



