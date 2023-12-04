import tensorflow as tf
import numpy as np
from skimage.io import imread
import os
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from preprocessing import masks_as_image

BATCH_SIZE = 32
IMG_SCALING = (3, 3) 

def IoU(y_true, y_pred, eps=1e-6):
    y_true = K.cast(y_true, dtype='float32') 
    y_pred = K.cast(y_pred, dtype='float32') 

    if K.max(y_true) == 0.0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, dtype='float32') 
    y_pred = K.cast(y_pred, dtype='float32') 

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, in_df, dir, batch_size=BATCH_SIZE, augmentation=True):
        self.all_batches = list(in_df.groupby('ImageId'))
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.dir = dir

        self.datagen_dict = dict(
            horizontal_flip=True,
            vertical_flip = True,
            )

        self.image_datagen = ImageDataGenerator(**self.datagen_dict)
        self.mask_datagen = ImageDataGenerator(**self.datagen_dict)

    def __len__(self):
        return int(np.ceil(len(self.all_batches) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.all_batches[idx * self.batch_size:(idx + 1) * self.batch_size]

        out_rgb = []
        out_mask = []

        for c_img_id, c_masks in batch_pairs:
            seed = np.random.choice(range(9999))
            
            rgb_path = os.path.join(self.dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)

            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

            if self.augmentation:
                c_img = self.image_datagen.random_transform(c_img, seed=seed)
                c_mask = self.mask_datagen.random_transform(c_mask, seed=seed)

            out_rgb += [c_img]
            out_mask += [c_mask]
        
        return np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)