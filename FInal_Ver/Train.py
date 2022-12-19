from data_preprocess import x_train1, x_train2, labels_train
from Utils import facemodel
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


class lossStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, eopch, logs={}):
        if (logs.get('loss') < 0.003):
            #print("\n----reach 60% accuracy, stop training----")
            self.model.stop_training = True


lossStop = lossStop()

def train():

    BATCH_SIZE = 16
    EPOCH = 1000

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
                tf.keras.callbacks.ModelCheckpoint(filepath='siamese_resnet152_221104_best_ours.h5',
                                                monitor='loss',
                                                save_best_only=True),
                lossStop]


    history = facemodel.fit([x_train1, x_train2], labels_train,callbacks = callbacks ,epochs = EPOCH, batch_size = BATCH_SIZE)

    facemodel.save('siamese_resnet152_Hun.h5')
    facemodel.save_weights('siamese_resnet152_weight_Hun.h5')


'''
#정확도 시각화
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy & Loss')
plt.legend(['accuracy', 'loss'], loc='upper right')
fig2 = plt.gcf()
fig2.savefig('siamese_resnet152_221104_ours.png')
'''