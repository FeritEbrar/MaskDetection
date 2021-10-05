"""
1) Libraries
2) File Operations-Data Preprocessing
3) Visualization
4) Training
5) Evaluation
6) Make Real Time Detection on MaskDetectionRealTime.py
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

path = "data"
categories = os.listdir(path)

noOfClasses = len(categories)

images = []
classes = []

for c in categories:
    image = os.listdir(path + "\\" + str(c))
    for i in image:
        img = cv2.imread(path + "\\" + str(c) + "\\" + str(i))
        img = cv2.resize(img, (32,32))
        images.append(img)
        classes.append(c)
        


images = np.array(images)
classes = [1 if i == "with_mask" else 0 for i in classes]
classes = np.array(classes)

print(len(images))
print(len(classes))

print(images.shape)
print(classes.shape)

x_train, x_test, y_train, y_test = train_test_split(images, classes, test_size= 0.2, random_state = 11)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 11)

print(f"Shape of x_train : {x_train.shape}\nShape of x_test : {x_test.shape}\nShape of x_validation : {x_validation.shape}\n")

# Visualization of countplot
# =============================================================================
# fig, axes = plt.subplots(3,1,figsize=(7,7))
# fig.subplots_adjust(hspace = 0.5)
# sns.countplot(y_train, ax = axes[0])
# axes[0].set_title("y_train")
#   
# sns.countplot(y_test, ax = axes[1])
# axes[1].set_title("y_test")
#   
# sns.countplot(y_validation, ax = axes[2])
# axes[2].set_title("y_validation")
# =============================================================================

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255 # normalization
    
    return img

# Deneme
# =============================================================================
# idx = 311
# img = preProcess(x_train[idx])
# img = cv2.resize(img,(150,150))
# cv2.imshow("Preprocess ",img)
# =============================================================================

# Preporcessi bütün veriye uygulamak
x_train = np.array(list(map(preProcess, x_train))) # map verilen fonksiyonu alır x_train e uygular
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
x_validation = x_validation.reshape(-1, 32, 32, 3)

dataGen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

dataGen.fit(x_train)

# OHE
y_train = to_categorical(y_train, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)

# CNN Architecture
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5,5), input_shape=(32,32,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding ='same'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = "binary_crossentropy", optimizer = ("Adam"), metrics = ["accuracy"])


batch_size = 64
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                           validation_data=(x_validation, y_validation),
                           epochs=40, steps_per_epoch = x_train.shape[0]/batch_size, shuffle=1)

# Write Model
open("face_detection_feg", "w").write(model.to_json())
model.save_weights("weights_face_detection.h5")

# Evaluate

hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()


score = model.evaluate(x_test, y_test, verbose = 1) # verbose : göster
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_validation, axis = 1)
cm = confusion_matrix(y_true, y_pred_class)
f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix")
plt.show()












