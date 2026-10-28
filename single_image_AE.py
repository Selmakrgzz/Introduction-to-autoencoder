import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from keras.models import Sequential

np.random.seed(42)

SIZE = 256

img_data = []

img = cv2.imread('../images/nemo.png',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (SIZE, SIZE))
img_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255

model = Sequential()
# Conv2D extracts the features
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape = (SIZE, SIZE, 3)))
# MaxPool2D shrinks the image
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(8, (3,3), activation='relu', padding='same'))

model.add(MaxPool2D((2,2), padding='same')) # bottleneck

model.add(Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same'))
model.add(Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same'))
model.add(Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same'))
model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
model.summary()

model.fit(img_array, img_array, epochs=500, shuffle=True)

pred = model.predict(img_array)
plt.imshow(pred[0])
plt.savefig('reconstruction.png')