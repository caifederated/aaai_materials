import os
import numpy as np
import tensorflow as tf

from PIL import Image

from model import Model


IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('data', 'celeba', 'data', 'raw', 'img_align_celeba')
NUM_CLASSES = 2

class ClientModel(Model):
    def __init__(self, seed, lr, train_bs=10, test_bs=16, input_shape=None, optimizer=None, sem=False):
        super(ClientModel, self).__init__(seed, lr, train_bs, test_bs, input_shape, optimizer, sem)
        
    def create_model(self):
        return self.create_CNNmodel()

    def create_CNNmodel(self):
        img_inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        out = img_inputs
        for _ in range(4):
            conv = tf.keras.layers.Conv2D(32, 3, padding='same')
            out = conv(out)
            bn = tf.keras.layers.BatchNormalization(trainable=True)
            out = bn(out)
            mp = tf.keras.layers.MaxPool2D(2, 2, padding='same')
            out = mp(out)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(out)

        model = tf.keras.Model(img_inputs, out)

        self.set_model_trainable_compile(model)
        return model

    def create_CNN_ma_model(self):
        img_inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        out = img_inputs
        for _ in range(4):
            conv = tf.keras.layers.Conv2D(32, 3, padding='same')
            out = conv(out)
            bn = tf.keras.layers.BatchNormalization(trainable=True)
            out = bn(out)
            mp = tf.keras.layers.MaxPool2D(2, 2, padding='same')
            out = mp(out)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))

        inner_model = tf.keras.Model(img_inputs, out)

        dense_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  
        ])

        out_model = tf.keras.models.Sequential([
          img_inputs,
          inner_model,
          dense_model
        ])

        self.set_model_trainable_compile(out_model)

        return inner_model, dense_model, out_model
    
    def create_normal_densemodel(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),            
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        self.set_model_trainable_compile(model)

        return model        

    def process_x(self, raw_x_batch):
      if isinstance(raw_x_batch[0], str):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch) / 255.0
      else:
        x_batch = raw_x_batch
      return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)
    
    def set_model_trainable_compile(self, model, trainable_bool=True):
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        
        return