import cv2 as cv
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import random
from keras.utils import to_categorical


def image_read_resize(file_dir):
  img = cv.imread(file_dir)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  img = cv.resize(img, (150, 150))
  return img

train_images = []
train_labels = []
test_images = []
test_labels = []

classes = ['rock', 'scissors', 'paper']

# creating train dataset

rock_dir = os.path.join('./tmp/rps/rock')
paper_dir = os.path.join('./tmp/rps/paper')
scissors_dir = os.path.join('./tmp/rps/scissors')

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

rocks = [os.path.join(rock_dir, fname) 
                for fname in rock_files]
papers = [os.path.join(paper_dir, fname) 
                for fname in paper_files]
scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files]

for i in range(len(rocks)):
  img_r = image_read_resize(rocks[i])
  train_images.append(img_r)
  train_labels. append(0)
  img_s = image_read_resize(scissors[i])
  train_images.append(img_s)
  train_labels.append(1)
  img_p = image_read_resize(papers[i])
  train_images.append(img_p)
  train_labels.append(2)

dataset = list(zip(train_images, train_labels))

random.shuffle(dataset)

train_images, train_labels = zip(*dataset)

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(classes[train_labels[i]])

# plt.show()

train_images = np.array(train_images)
train_labels = to_categorical(train_labels, num_classes=3)


# creating test dataset

rock_dir = os.path.join('./tmp/rps-test-set/rock')
paper_dir = os.path.join('./tmp/rps-test-set/paper')
scissors_dir = os.path.join('./tmp/rps-test-set/scissors')

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

rocks = [os.path.join(rock_dir, fname) 
                for fname in rock_files]
papers = [os.path.join(paper_dir, fname) 
                for fname in paper_files]
scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files]

for i in range(len(rocks)):
  img_r = image_read_resize(rocks[i])
  test_images.append(img_r)
  test_labels.append(0)
  img_s = image_read_resize(scissors[i])
  test_images.append(img_s)
  test_labels.append(1)
  img_p = image_read_resize(papers[i])
  test_images.append(img_p)
  test_labels.append(2)

dataset = list(zip(test_images, test_labels))

random.shuffle(dataset)

test_images, test_labels = zip(*dataset)

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel(classes[test_labels[i]])

# plt.show()

test_images = np.array(test_images)
test_labels = to_categorical(test_labels, num_classes=3)

# create model

model = tf.keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(3, activation='softmax')
    ])

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data = (test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)

print(loss)
print(accuracy)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()



predict_images = []

val_dir = os.path.join('./tmp/rps-validation')

val_files = os.listdir(val_dir)

val_image_dir = [os.path.join(val_dir, fname) 
                for fname in val_files]

for i in range(len(val_image_dir)):
    img = image_read_resize(val_image_dir[i])
    predict_images.append(img)

random.shuffle(predict_images)

predict_images = np.array(predict_images)

predictions = model.predict(predict_images)

predict_labels = [np.argmax(prediction) for prediction in predictions]

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[predict_labels[i]])

plt.show()
