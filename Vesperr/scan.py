import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from PIL import Image
from sklearn.model_selection import train_test_split


normal_cells=os.listdir('/Data/Normal')

print(normal_cells[0:5])
print(normal_cells[-5:])


tumor_cells=os.listdir('Data/Tumor')

print(tumor_cells[0:5])
print(tumor_cells[-5:])


print('Length of the Normal Brain Cells: ',len(normal_cells))
print('Length of the Cancurus Brain Cells: ',len(tumor_cells))

normal_label=[0]*98
tumor_label=[1]*155

print('Normal Cell labels: ',len(normal_label))
print('Tumor Cell labels: ',len(tumor_label))

print('With Normal labels: ',normal_label[0:5])
print('With Tumor labels: ',tumor_label[0:5])



labels=normal_label+tumor_label

print('Total labels are: ',len(labels))

print(labels[0:5])
print(labels[-5:])

type(labels)

"""---------

## Data Visualization 📊

### a) Normal Cell image
"""

nor_img=mpimg.imread('/content/drive/MyDrive/Colab Notebooks/Data/Normal/N_1_BR_.jpg')

plt.imshow(nor_img)

"""### b) Tumour Cell image"""

nor_img=mpimg.imread('/content/drive/MyDrive/Colab Notebooks/Data/Tumor/Copy of G_1.jpg')

plt.imshow(nor_img)

"""### c) See the Distribution of the Labeled column"""

import seaborn as sn


sn.countplot(x=labels)

"""------

## Image Preprocessing Steps
"""

normal_path=('/content/drive/MyDrive/Colab Notebooks/Data/Normal/')
data=[]

for img_file in normal_cells:
    image=Image.open(normal_path + img_file)
    image=image.resize((128,128))
    image=image.convert('RGB')
    image=np.array(image)
    data.append(image)

tumor_path=('/content/drive/MyDrive/Colab Notebooks/Data/Tumor/')

for img_file in tumor_cells:
    image=Image.open(tumor_path + img_file)
    image=image.resize((128,128))
    image=image.convert('RGB')
    image=np.array(image)
    data.append(image)

type(data)

"""**total length of the data**"""

len(data)

"""**Checking the first image**"""

data[0]

type(data[0])

"""**Checking the shape of the single image**"""

data[0].shape

"""----------

## Converting data and labels into numpy array
"""

X=np.array(data)
Y=np.array(labels)

type(X)

type(Y)

print(X.shape)
print(Y.shape)

"""---------

## Train Test Split
"""

X = X[:253]  # Truncate X to match the length of Y

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=100)

print(X.shape,X_train.shape,X_test.shape)

"""------

## Scaling the data
"""

X_train=X_train/255
X_test=X_test/255

X_train[0]

"""-------------

## Model Building
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout

"""### a) Model Building"""

num_of_classes=2

model=Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_of_classes, activation='sigmoid'))

"""### b) Compiling the model"""

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""### c) Training the model"""

history=model.fit(X_train,Y_train, epochs=50, validation_split=0.1, verbose=1)

"""### d) Model Evaluation"""

model.evaluate(X_test,Y_test)

"""#### Accuracy is 86%

### e) Learning Curve

**i) Accuracy Curve**
"""

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')

plt.legend()
plt.show()

"""**ii) Loss Curve**"""

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')

plt.legend()
plt.show()

"""----

## Prediction Function

### a) For Normal Cell
"""

input_image_path=input('Enter the path of the image: ')

input_image=cv2.imread(input_image_path)

plt.imshow(input_image)
plt.show()


# Ensure the shape matches the expected input shape of your model
input_image_reshape=cv2.resize(input_image,(128,128))


# Normalize the pixel values
image_normalized=input_image_reshape/255


# Reshape for model prediction
img_reshape=np.reshape(image_normalized,(1,128,128,3))


# Make Predictions
input_prediction=model.predict(img_reshape)

# Display the prediction Probabilities
print('Prediction Probabilities are:  ',input_prediction)


# Get the Predicted Label
input_pred_label=np.argmax(input_prediction)


if input_pred_label ==1:
    print('Tumor Cell')
else:
    print('Normal Cell')

"""### b) For Tumor Cell"""

input_image_path=input('Enter the path of the image: ')

input_image=cv2.imread(input_image_path)

plt.imshow(input_image)
plt.show()


# Ensure the shape matches the expected input shape of your model
input_image_reshape=cv2.resize(input_image,(128,128))


# Normalize the pixel values
image_normalized=input_image_reshape/255


# Reshape for model prediction
img_reshape=np.reshape(image_normalized,(1,128,128,3))


# Make Predictions
input_prediction=model.predict(img_reshape)

# Display the prediction Probabilities
print('Prediction Probabilities are:  ',input_prediction)


# Get the Predicted Label
input_pred_label=np.argmax(input_prediction)


if input_pred_label ==1:
    print('Tumor Cell')
else:
    print('Normal Cell')

"""-------

## Saving the Model
"""

model.save('/kaggle/working/tumor_detection.h5')

"""-------"""