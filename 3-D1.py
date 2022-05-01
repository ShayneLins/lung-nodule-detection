
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization,AveragePooling2D,Activation
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

working_path = "D:/BaiduYunDownload/final/"
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
img_rows = 512
img_cols = 512
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
def dice(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_true_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
# def get_unet():
#     inputs = Input((1,img_rows, img_cols))
#     conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Activation('relu')(conv1)
#     print(conv1.shape)
#     #conv1 = Conv2D(1, 1, 1)(conv1)
#     conv2 = Conv2D(32, 1, 1,  border_mode='same')(conv1)
#     print('conv2',conv2.shape)
#     conv2 = BatchNormalization()(conv2)
#     print(conv2.shape)

# #     #conv2 = Conv2D(1, 1, 1)(conv2)
# #     up1 = concatenate([conv1, conv2], axis=1)
# #     conv3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up1)
# #     conv3 = BatchNormalization()(conv3)
# #     #conv3 = Conv2D(1, 1, 1)(conv3)
# #     up2 = concatenate([up1, conv3], axis=1)
# #     conv4 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up2)
# #     conv4 = BatchNormalization()(conv4)
# #     #conv3 = Conv2D(1, 1, 1)(conv3)
# #     up3 = concatenate([up2, conv4], axis=1)

#     conv9 = Conv2D(1, 1, 1, activation='sigmoid')(conv2)
#     model = Model(input=inputs, output=conv9)
#     model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
#     return model
def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv0 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    
    conv1 = Conv2D(32, 3, 3,border_mode='same')(pool0)
    bn1 = BatchNormalization()(conv1)
    re1 = Activation('relu')(bn1)
    con1 = Conv2D(32, 1, 1, border_mode='same')(re1)
    
    conv2 = Conv2D(32, 3, 3, border_mode='same')(con1)
    up1 = concatenate([conv2, conv1], axis=1)
    bn2 = BatchNormalization()(up1)
    re2 = Activation('relu')(bn2)
    con2 = Conv2D(32, 1, 1, border_mode='same')(re2)
    
    conv3 = Conv2D(32, 3, 3, border_mode='same')(con2)
    up2 = concatenate([up1, conv3], axis=1)
    bn3 = BatchNormalization()(up2)
    re3 = Activation('relu')(bn3)
    con3 = Conv2D(32, 1, 1, border_mode='same')(re3)
    
    conv4 = Conv2D(32, 3, 3, border_mode='same')(con3)
    up3 = concatenate([up2, conv4], axis=1)
    bn4 = BatchNormalization()(up3)
    re4 = Activation('relu')(bn4)
    con4 = Conv2D(32, 1, 1, border_mode='same')(re4)
    
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(con4)
    up4 = concatenate([up3, conv5], axis=1)
    up5 = concatenate([UpSampling2D(size=(2, 2))(up4), conv0], axis=1)
    
    con5 = Conv2D(32, 1, 1, border_mode='same')(up5)
    pool1 = AveragePooling2D(pool_size=(2, 2))(con5)
    
    print('one block')
    
    conv6 = Conv2D(32, 3, 3,border_mode='same')(pool1)
    bn5 = BatchNormalization()(conv6)
    re5 = Activation('relu')(bn5)
    con6 = Conv2D(32, 1, 1, border_mode='same')(re5)
    
    conv7 = Conv2D(32, 3, 3, border_mode='same')(con6)
    up6 = concatenate([conv7, conv6], axis=1)
    bn6 = BatchNormalization()(up6)
    re6 = Activation('relu')(bn6)
    con7 = Conv2D(32, 1, 1, border_mode='same')(re6)
    
    conv8 = Conv2D(32, 3, 3, border_mode='same')(con7)
    up7 = concatenate([up6, conv8], axis=1)
    bn7 = BatchNormalization()(up7)
    re7 = Activation('relu')(bn7)
    con8 = Conv2D(32, 1, 1, border_mode='same')(re7)
    
    conv9 = Conv2D(32, 3, 3, border_mode='same')(con8)
    up8 = concatenate([up7, conv9], axis=1)
    bn8 = BatchNormalization()(up8)
    re8 = Activation('relu')(bn8)
    con9 = Conv2D(32, 1, 1, border_mode='same')(re8)
    
    conv10 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(con9)
    up9 = concatenate([up8, conv10], axis=1)
    up10 = concatenate([UpSampling2D(size=(2, 2))(up9), conv0], axis=1)
    
    con10 = Conv2D(32, 1, 1, border_mode='same')(up10)
    pool2 = AveragePooling2D(pool_size=(2, 2))(con10)

    print('two block')
    
    conv11 = Conv2D(32, 3, 3,border_mode='same')(pool2)
    bn9 = BatchNormalization()(conv11)
    re9 = Activation('relu')(bn9)
    con11 = Conv2D(32, 1, 1, border_mode='same')(re9)
    
    conv12 = Conv2D(32, 3, 3, border_mode='same')(con11)
    up11 = concatenate([conv12, conv11], axis=1)
    bn10 = BatchNormalization()(up11)
    re10 = Activation('relu')(bn10)
    con12 = Conv2D(32, 1, 1, border_mode='same')(re10)
    
    conv13 = Conv2D(32, 3, 3, border_mode='same')(con12)
    up12 = concatenate([up11, conv13], axis=1)
    bn11 = BatchNormalization()(up12)
    re11 = Activation('relu')(bn11)
    con13 = Conv2D(32, 1, 1, border_mode='same')(re11)
    
    conv14 = Conv2D(32, 3, 3, border_mode='same')(con13)
    up13 = concatenate([up12, conv14], axis=1)
    bn12 = BatchNormalization()(up13)
    re12 = Activation('relu')(bn12)
    con14 = Conv2D(32, 1, 1, border_mode='same')(re12)
    
    conv15 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(con14)
    up14 = concatenate([up13, conv15], axis=1)
    up15 = concatenate([UpSampling2D(size=(2, 2))(up14), conv0], axis=1)
    
    conv16 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up15)
    conv17 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv16)
    conv18 = Conv2D(1, 1, 1, activation='sigmoid')(conv17)
#     print(conv9.shape)
    model = Model(input=inputs, output=conv18)
    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)
    imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    mean = np.mean(imgs_train)  # mean for data centering
#     print(imgs_train.shape)
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
#     model.save_weights('./densenet3.hdf5')

    model_checkpoint = ModelCheckpoint('densenet3.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./densenet3.hdf5')        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
#     model.fit(imgs_train, imgs_mask_train, batch_size=2, epoch=20, verbose=1, shuffle=True,
#               callbacks=[model_checkpoint])

    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=12, verbose=1, shuffle=True,callbacks=[model_checkpoint])
    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./densenet3.hdf5')
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save(working_path+'1DmasksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    ACC = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    for i in range(num_test):
        ACC+=dice(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    ACC/=num_test
    print("ACC : ",ACC)
if __name__ == '__main__':
    train_and_predict(True)