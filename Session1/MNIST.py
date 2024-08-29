import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
# output (32, 28, 28, 1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
# output (32, 14， 14)
model.add(layers.MaxPooling2D((2, 2)))
# output 使用的卷积核位（3，3，32）得到的输出位（64， 14， 14）
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# output （64， 7， 7）
model.add(layers.MaxPooling2D(2, 2))
# output （64，7， 7）
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# output （64*7*7）
model.add(layers.Flatten())
# 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 步骤 3: 编译和训练模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 步骤 4: 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# 313/313 [==============================] - 1s 3ms/step - loss: 0.0307 - accuracy: 0.9911
# Test accuracy: 0.9911