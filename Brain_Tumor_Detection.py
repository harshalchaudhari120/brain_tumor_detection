#!/usr/bin/env python
# coding: utf-8

# #### Brain Tumor Detection using base model ResNet50V2  With Accuracy~98%

# #### Data Uploading

# In[1]:


import os
import numpy as np
import random
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance


test_dir = "C:/MIT/brain_tumor_detection dataset/Testing/"
train_dir = "C:/MIT/brain_tumor_detection dataset/Training/"

# Collecting image paths and labels
train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    if os.path.isdir(label_dir):
        for image in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image)
            train_paths.append(image_path)
            train_labels.append(label)

# Shuffling the data
train_paths, train_labels = shuffle(train_paths, train_labels)

IMAGE_SIZE = 128

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image

def open_images(paths):
    images = [augment_image(load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))) for path in paths]
    return np.array(images)

images = open_images(train_paths[:10])
labels = train_labels[:10]

plt.figure(figsize=(12, 6))

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(labels[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for image in os.listdir(test_dir+label):
        test_paths.append(test_dir+label+'/'+image)
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)


# In[2]:


unique_labels=os.listdir(train_dir)

def encode_label(labels):
    encoded=[]
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_label(label):
    return unique_labels[label]

def data_generator(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels


# #### Model Building,Deployment

# In[3]:


from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy



IMAGE_SIZE = 128
NUM_CLASSES = len(unique_labels)  # You need to define `unique_labels` before using this

# Loading the ResNet50V2 model with pre-trained weights
base_model = ResNet50V2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Setting all layers in the base model to non-trainable
for layer in base_model.layers:
    layer.trainable = False

# Setting the last few layers in the base model to trainable
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Creating a sequential model
model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.5),  
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),   
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

# Using the Adam optimizer with a lower learning rate
# Lower learning rate for fine-tuning
optimizer = Adam(lr=0.0001) 
# Compiling the model with Adam optimizer and custom learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])
#summary of the model architecture
model.summary()


# In[4]:


batch_size = 20
steps_per_epoch = int(len(train_paths) / batch_size)
epochs = 6

# Training the model using the data generator
history = model.fit(data_generator(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps_per_epoch)


# In[5]:


plt.figure(figsize=(10, 6))
plt.plot(history.history['sparse_categorical_accuracy'], 'g-', linewidth=2, label='Accuracy')  # Changed color and label
plt.plot(history.history['loss'], 'r-', linewidth=2, label='Loss')  # Changed color and label
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.xticks(range(epochs))  # Use `range` directly
plt.legend(loc='upper left')
plt.show()

 


# #### Evaluating the Model

# In[6]:


from tqdm import tqdm
batch_size = 32
steps = int(len(test_paths) / batch_size)
y_pred = []
y_true = []

for x, y in tqdm(data_generator(test_paths, test_labels, batch_size=batch_size, epochs=1), total=steps):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    
    y_pred.extend(pred)  # Extend the predicted labels list
    y_true.extend(y)     # Extend the true labels list

# Convert the predicted and true labels to human-readable labels
y_pred_labels = [decode_label(label) for label in y_pred]
y_true_labels = [decode_label(label) for label in y_true]


# In[7]:


from sklearn.metrics import classification_report
print(classification_report(y_true_labels, y_pred_labels))

