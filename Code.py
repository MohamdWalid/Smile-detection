

import numpy as np
import pandas as pd
from math import sqrt


images_no_smile = []
y_no_smile = []

images_smile = []
y_smile = []
for i in range(1,13300):
        # Data contain 7380 image
        if (len(images_smile) == 3690  and len(images_no_smile) == 3690):
            break
        # no smile
        if(len(images_no_smile)<3690):
          image = cv2.imread(fr'C:\Users\Walid\OneDrive\Desktop\Smile Detection\negatives_No_Smile\{i}.jpg',0)
          try:
            shape=image.shape
            images_no_smile.append(image.ravel())
            y_no_smile.append(0)
          except:
            pass

        # Smile
        if (len(images_smile) < 3690):
          image = cv2.imread(fr'C:\Users\Walid\OneDrive\Desktop\Smile Detection\positives_Smile\{i}.jpg', 0)
          try:
            shape = image.shape
            images_smile.append(image.ravel())
            y_smile.append(1)
          except:
            pass

# make data in one list
images = []
label = []
for i in range(0,3690):
    images.append(images_smile[i])
    label.append(y_smile[i])
    images.append(images_no_smile[i])
    label.append(y_no_smile[i])

images = np.array(images)
label = np.array(label).reshape(-1,1)
Data = np.concatenate([images,label],axis=1)

x_train = Data[:7000,:]
x_test = Data[7000:,:]
y_pred = []

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# manual
k = 7
for i in range(len(x_test)):
   print(i)
   y_pred.append(predict_classification(x_train, x_test[i], k))

y_test = x_test[:,-1:]
y_pred = np.array(y_pred).reshape(-1,1)
# accuracy = 0.77
print(f"manual KNN Accuracy: {np.count_nonzero(y_test==y_pred) / len(y_test)} ")

# Sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(x_train[:,:-1], x_train[:,-1:])
y_pred_test = clf.predict(x_test[:,:-1])
# accuracy = 0.77
print(f"Sklearn KNN Accuracy: {accuracy_score(x_test[:,-1:], y_pred_test)}")