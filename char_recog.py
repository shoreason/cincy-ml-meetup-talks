import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm

digits =  datasets.load_digits()

# get me SVM classifer
clf = svm.SVC(gamma=0.001, C=100)
# get everything except the last 10

x, y = digits.data[:-10], digits.target[:-10]
clf.fit(x, y)

print("amount of data: ", len(x))

test = digits.data[-2]
test = np.array(test).reshape(1, -1)
print('Prediction: ', clf.predict(test))

print(digits.images[-2])
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
