from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

from joblib import dump, load
dump(clf, 'filename.joblib')
# clf = load('filename.joblib')
# clf2 = load ('e07_002_001_0347.adc')


# dump and load in accept file like objects
# with open(filename, 'wb') as fo:
     # joblib.dump(to_persist, fo)
# with open(filename, 'rb') as fo:
     # joblib.load(fo)

import numpy as np
import scipy as sp
data = np.fromfile("../e07_002_001_0349.adc")
# data2 = sp.read(r"C:\Users\Magomed\Desktop\ML\UKATrialCompus\audio\002\001\a_002_001_0349.wav")
print (".data:")
print (data)
print (".shape:")
print (data.shape)
print (".dim:")
print (data.ndim)
