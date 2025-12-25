from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import math
import numpy as np
import os
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:,1] * y_pred[:,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:,1] * y_pred[:,1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        if y_pred[i,1] >= 0.5:
            y_pred[i,1]=1
            y_pred[i,0]=0
        else:
            y_pred[i,1]=0
            y_pred[i,0]=1

    TP=sum(y_true[:,1] * y_pred[:,1])
    TN=sum(y_true[:,0] * y_pred[:,0])
    FP=sum(y_true[:,0] * y_pred[:,1])
    FN=sum(y_true[:,1] * y_pred[:,0])


    return TP, FP, TN, FN


def CNN(number_filters,filter_size,number_columns,length,lr=1e-3):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(number_filters, kernel_size=filter_size, padding='same',strides=1, input_shape=(length, number_columns),activation='relu'),
    tf.keras.layers.Conv1D(number_filters, kernel_size=filter_size, padding='same',strides=1, input_shape=(length, number_filters),activation='relu'),
    tf.keras.layers.MaxPooling1D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5),
                  loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    return model

def CNN_2single_block(number_filters,filter_size,number_columns,length,lr=1e-3):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(number_filters, kernel_size=filter_size, padding='same',strides=1, input_shape=(length, number_columns),activation='relu'),
    tf.keras.layers.MaxPooling1D(2,2),
    
    tf.keras.layers.Conv1D(number_filters, kernel_size=filter_size, padding='same',strides=1, input_shape=(length, number_columns),activation='relu'),
    tf.keras.layers.MaxPooling1D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5),
                  loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    return model

def AlexNet(nb_classes,input_shape):
    input_ten = Input(shape=input_shape)
    #1
    x = tf.keras.layers.Conv1D(filters=96,kernel_size=(11,11),strides=(4,4),activation='relu')(input_ten)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=(3,3),strides=2)(x)
    #2
    x = tf.keras.layers.Conv1D(filters=256,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=(3,3),strides=2)(x)
    #3
    x = tf.keras.layers.Conv1D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(x)
    #4
    x = tf.keras.layers.Conv1D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(x)
    #5
    x = tf.keras.layers.Conv1D(filters=256,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=(3,3),strides=2)(x)
    #FC
    x = tf.keras.layers.Flatten()(x)
    x = Dense(4096,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(4096,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_ten = tf.keras.layers.Dense(nb_classes,activation='softmax')(x)
    model = Model(input_ten,output_ten)
    return model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def AlexNet1D():
    # 模型构建
    model = keras.models.Sequential([
        layers.Conv1D(filters=96, kernel_size=11, strides=4, activation='relu'),
        layers.MaxPool1D(pool_size=3, strides=2),
        layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPool1D(pool_size=3, strides=2),
        layers.Conv1D(filters=384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv1D(filters=384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool1D(pool_size=3, strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')],
        name='AlexNet')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5),
                loss='categorical_crossentropy', metrics=['accuracy',f1_m])
    
    return model


net = keras.models.Sequential([
    layers.Conv1D(filters=96, kernel_size=11, strides=4, activation='relu'),
    layers.MaxPool1D(pool_size=3, strides=2),
    layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
    layers.MaxPool1D(pool_size=3, strides=2),
    layers.Conv1D(filters=384, kernel_size=3, padding='same', activation='relu'),
    layers.Conv1D(filters=384, kernel_size=3, padding='same', activation='relu'),
    layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
    layers.MaxPool1D(pool_size=3, strides=2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')],
    name='AlexNet')
x = tf.random.uniform((1,327,14))
y = net(x)

net.summary()
# x = tf.random.uniform((1, 227, 227, 1))
# y = net(x)

# net.summary()

