#!/usr/bin/env python

import os, random, glob, keras
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import pandas as pd
from data_generator import imageLoader
from unet_2d_model import simple_2d_unet_model
from cal_dice_coef import cal_dice_coef

os.environ["SM_FRAMEWORK"] = "tf.keras"


LR     = 0.0001
optim  = keras.optimizers.Adam(LR)
nEpoch = 100
seed   = 42
np.random.seed = seed

IMG_WIDTH      = 512 # 128
IMG_HEIGHT     = 512 # 128
IMG_CHANNELS   = 1   # 3

batch_size     = 2


# put your nifti data in the following directory.
# the file should be ended with .nii.gz 

dataPath       = '/*/train_test_val_sv_fat' 
train_img_dir  =  dataPath +  '/train/image' # one image in one directory: /*/train_test_val_sv_fat/train/image/subject/sub1_image.nii.gz
train_mask_dir =  dataPath + '/train/mask'   # /*/train_test_val_sv_fat/train/mask/subject/sub1_msk.nii.gz

test_img_dir   =  dataPath +  '/test/image'  
test_mask_dir  =  dataPath + '/test/mask'   

val_img_dir    =  dataPath + '/val/image' 
val_mask_dir   =  dataPath + '/val/mask' 


train_img_list  = sorted(glob.glob(train_img_dir+os.sep+'*'+os.sep+'*'))
train_mask_list = sorted(glob.glob(train_mask_dir+os.sep+'*'+os.sep+'*'))

test_img_list   = sorted(glob.glob(test_img_dir+os.sep+'*'+os.sep+'*'))
test_mask_list  = sorted(glob.glob(test_mask_dir+os.sep+'*'+os.sep+'*'))

val_img_list    = sorted(glob.glob(val_img_dir+os.sep+'*'+os.sep+'*'))
val_mask_list   = sorted(glob.glob(val_mask_dir+os.sep+'*'+os.sep+'*'))

train_img_datagen  = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size,IMG_WIDTH,IMG_HEIGHT)

test_img_datagen   = imageLoader(test_img_dir, test_img_list, 
                                test_mask_dir, test_mask_list, batch_size,IMG_WIDTH,IMG_HEIGHT)


model = simple_2d_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_vs_fat.hdf5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

#Fit the model 
steps_per_epoch       = len(train_img_list)//batch_size
test_steps_per_epoch  = len(val_img_list)//batch_size

history = model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs= nEpoch, #100,
          verbose=1,
          validation_data=test_img_datagen,
          validation_steps=test_steps_per_epoch,
          callbacks=callbacks)

outModel = 'model_2d_unet_' + str(nEpoch) + '_epochs'  + '.hdf5'  #'model_test.hdf5'

model.save(outModel)
# load the model 
model = tf.keras.models.load_model(outModel,compile=False)

model.summary()

# select 3 subjects randomly within each group
idx1 = random.randint(0, len(train_img_list))
idx2 = random.randint(0, len(test_img_list))
idx3 = random.randint(0, len(val_img_list))

print(train_img_list[idx1])
X_train       = nib.load(train_img_list[idx1])
X_train       = X_train.get_fdata()# np.squeeze(X_train.get_fdata())
X_train  = 255 * (X_train - X_train.min()) / (X_train.max() - X_train.min() + 0.001) 
X_train  = np.expand_dims(X_train, axis=0)
Y_train       = nib.load(train_mask_list[idx1])
Y_train       = np.round(np.squeeze(Y_train.get_fdata()))

print(test_img_list[idx2])
X_test        = nib.load(test_img_list[idx2])
X_test        = X_test.get_fdata()# np.squeeze(X_test.get_fdata())
X_test  = 255 * (X_test - X_test.min()) / (X_test.max() - X_test.min() + 0.001) 
X_test  = np.expand_dims(X_test, axis=0)
Y_test        = nib.load(test_mask_list[idx2])
Y_test        = np.round(np.squeeze(Y_test.get_fdata()))

X_val         = nib.load(val_img_list[idx3])
X_val         = X_val.get_fdata() # np.squeeze(X_val.get_fdata())
X_val  = 255 * (X_val - X_val.min()) / (X_val.max() - X_val.min() + 0.001) 
X_val  = np.expand_dims(X_val, axis=0)
Y_val         = nib.load(val_mask_list[idx3])
Y_val         = np.round(np.squeeze(Y_val.get_fdata()))

preds_train   = model.predict(X_train, verbose=1)
preds_test    = model.predict(X_test, verbose=1)
preds_val     = model.predict(X_val, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t   = (preds_val > 0.5).astype(np.uint8)
preds_test_t  = (preds_test > 0.5).astype(np.uint8)

plt.subplot(231)
plt.imshow(np.squeeze(preds_train_t), cmap='gray')
plt.title('Train: preds_train')
plt.subplot(234)
plt.imshow(Y_train, cmap='gray')
diceTrain = cal_dice_coef(np.squeeze(preds_train_t), Y_train)
diTrain   = str(diceTrain)
plt.title('Train: ground truth mask, diceCoef=' + diTrain[0:4])

plt.subplot(232)
plt.imshow(np.squeeze(preds_test_t), cmap='gray')
plt.title('Test:preds_test')
plt.subplot(235)
plt.imshow(Y_test, cmap='gray')
diceTest = cal_dice_coef(np.squeeze(preds_test_t), Y_test)
diTest   = str(diceTest)
plt.title('Test: ground truth mask,diceCoef='+diTest[0:4])

plt.subplot(233)
plt.imshow(np.squeeze(preds_val_t), cmap='gray')
plt.title('Val: preds validation image')
plt.subplot(236)
plt.imshow(Y_val, cmap='gray')
diceVal = cal_dice_coef(np.squeeze(preds_val_t), Y_val)
diVal   = str(diceVal)
plt.title('Val: ground truth mask,diceCoef='+diVal[0:4])
plt.show()


df_acc     =  outModel.split('.')[0] + '_model_fit_hisotry.csv'

col_names  = ['loss', 'accuracy', 'val_loss','val_accuracy']
df         = pd.DataFrame(columns = col_names)
df['loss'] = history.history['loss']
df['accuracy'] = history.history['accuracy']
df['val_loss'] = history.history['val_loss']
df['val_accuracy'] = history.history['val_accuracy']

if os.path.exists(df_acc):
    df1     = pd.read_csv(df_acc)    
    df_comb = pd.concat([df, df1], ignore_index=True, sort=False)
    str1    = 'rm ' + df_acc
    # os.system(str1)
    df_comb.to_csv(df_acc)
else:
    df.to_csv(df_acc)


