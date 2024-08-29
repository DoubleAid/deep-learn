import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.applications import ResNet50

import numpy as np

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    zoom_range=0.2,
)

def try_to_use_gpu():
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

def image_processing_generator(images, labels, batch_size):
    num_samples = images.shape[0]
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    while True:
        for offset in range(0, num_samples, batch_size):
            # 计算当前批次的结束索引
            batch_end = min(offset + batch_size, num_samples)
            batch_images = images[offset:batch_end]
            batch_labels = labels[offset:batch_end]

            # 调整图像大小并复制通道
            batch_images = np.array([tf.image.resize(img, [224, 224]).numpy() for img in batch_images])
            batch_images = np.repeat(batch_images, 3, axis=-1)

            # 归一化图像数据
            batch_images = batch_images.astype('float32') / 255.0

            for i in range(batch_images.shape[0]):
                batch_images[i] = datagen.random_transform(batch_images[i])

            # 使用to_categorical确保标签是正确的one-hot编码
            batch_labels = to_categorical(batch_labels, num_classes=10)

            yield batch_images, batch_labels

def load_data1():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("PRE Train images shape:", train_images.shape)
    print("PRE Test images shape:", test_images.shape)
    print("PRE Train labels shape:", train_labels.shape)
    print("PRE Test labels shape:", test_labels.shape)

    train_images = np.repeat(train_images[..., np.newaxis], 3, axis=3)
    test_images = np.repeat(test_images[..., np.newaxis], 3, axis=3)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # 数据预处理
    train_images = tf.image.resize(train_images, [224, 224])
    test_images = tf.image.resize(test_images, [224, 224])

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images.numpy(), train_labels, test_images.numpy(), test_labels)

def load_data2():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # 数据预处理
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels, test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_generator = image_processing_generator(train_images, train_labels, batch_size=32)
test_generator = image_processing_generator(test_images, test_labels, batch_size=32)

def get_model1():
    # accuracy 0.8855
    model1 = models.Sequential([
        Input(shape=(28, 28 ,1)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
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

        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model1

def get_model2():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

class TestAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs = None):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose = 0)
        print(f"\nTest loss: {test_loss}, test accuracy: {test_acc}")

if __name__ == "__main__":
    test_accuracy_callback = TestAccuracyCallback()

    model = get_model2()
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    steps_per_epoch = len(train_images) // 32
    validation_steps = len(test_images) // 32

    model.fit(train_generator, 
              steps_per_epoch=steps_per_epoch,
              epochs=8,
              validation_data=image_processing_generator(test_images, test_labels, batch_size=32),
              validation_steps=validation_steps,
              callbacks=[test_accuracy_callback])

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")