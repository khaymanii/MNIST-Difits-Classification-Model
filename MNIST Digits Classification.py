

# Importing the dependencies

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from cv2_plt_imshow import cv2_plt_imshow, plt_format
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix


# In[4]:


# Loading the MNIST data from keras.datasets

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


type(X_train)


# In[7]:


# Shape of numpy array

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[8]:


# printing the tenth image

print(X_train[10])


# In[9]:


# Displaying the image

plt.imshow(X_train[50])
plt.show()

# print the corresponding label

print(y_train[50])


# In[10]:


print(y_train.shape, y_test.shape)


# In[11]:


# Unique values in y_train

print(np.unique(y_train))

# Unique values in y_test

print(np.unique(y_test))


# In[12]:


# Scaling the values

X_train = X_train/255
X_test = X_test/255


# In[13]:


# Building the neural network

# Setting up the layers of the neural networks

model = keras.Sequential([
       keras.layers.Flatten(input_shape=(28,28)),
       keras.layers.Dense(50, activation='relu'),
       keras.layers.Dense(50, activation='relu'),
       keras.layers.Dense(10, activation='sigmoid')
])


# In[14]:


# compiling the Neural networks

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


# Training the neural network

model.fit(X_train, y_train, epochs=10)


# In[16]:


# Evaluation on test data accuracy

loss, accuracy = model.evaluate(X_test,y_test)
print(accuracy)


# In[17]:


# first data point in X_test

plt.imshow(X_test[0])
plt.show()


# In[18]:


print(y_test[0])


# In[19]:


y_pred = model.predict(X_test)


# In[20]:


print(y_pred[0])


# In[21]:


# converting the prediction probabilities to class label

label_for_first_image = np.argmax(y_pred[0])
print(label_for_first_image)


# In[23]:


# converting the prediction probabilities to class label for all test datapoints

y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels)


# In[24]:


# Building the confusion matrix

conf_mat = confusion_matrix(y_test, y_pred_labels)


# In[25]:


print(conf_mat)


# In[31]:


plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')


# In[ ]:




