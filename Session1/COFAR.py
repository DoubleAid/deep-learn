import tensorflow as tf

from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import Callback

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 如果有多个GPU，仅使用第一个GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # 设置内存增长模式，避免占用全部可用GPU内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 程序启动时需要设置GPU
        print(e)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# 归一化，使数据训练更加高效
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Training Data shape: ", train_images.shape)

# 准确度 0.5448
model1 = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', 
                  input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Dropout(0.5), 
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# 准确度 0.6079
model2 = models.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Dropout(0.2), 
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# 准确度 0.6208
model3 = models.Sequential([
    # 第一层卷积，输入层（32x32x3）
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                  input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # 第二层卷积
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Flatten层转换为一维数据，连接全连接层
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 准确度 0.7642
model4 = models.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(64, (3, 3), padding = 'same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model = model4

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    zoom_range=0.2,
)

class TestAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs = None):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose = 0)
        print(f"Test loss: {test_loss}, test accuracy: {test_acc}")

test_accuracy_callback = TestAccuracyCallback()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(datagen.flow(train_images, train_labels, batch_size=64), 
          epochs=8,
          validation_data=(test_images, test_labels),
          callbacks=[test_accuracy_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")