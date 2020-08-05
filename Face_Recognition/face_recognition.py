# Recognise Faces using some classification algorithm - like logistic, KNN, SVM etc.

# 1. Load the training data (numpy arrays of all the persons)
        # x- values are stored in the numpy array
        # y- values we need to assign for each person
# 2. Read a video stream using opencv
# 3. Extract faces out of it 
# 4. Use KNN to find the prediction of face (int)
# 5. Map the predicted if to name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np 
import cv2
import os

# KNN code
def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		ix = train[i, :-1]
		iy = train[i, -1]
		d = distance(test, ix)
		dist.append([d, iy])

	dk = sorted(dist, key=lambda x: x[0])[:k]
	labels = np.array(dk)[:, -1]

	output = np.unique(labels, return_counts=True)
	index = np.argmax(output[1])

	return output[0][index]

# Initialize Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'
face_data = [] # x values
label = [] # y values

class_id = 0 # Labels for the given file
names = {} # Mapping between id - name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # create a mapping between class_id and the label
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(label, axis = 0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels),axis = 1)
print(trainset.shape)

# Testing
while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        # Get the face region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # Predicted label (out)
        out = knn(trainset, face_section.flatten())

        # Display on the screen the name and rectangle around it 
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
