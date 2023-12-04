import numpy as np
from scipy import ndimage
from skimage.io import imread
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

IMG_SCALING = (3, 3)

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def split_mask(mask):
    threshold = 0.6
    threshold_obj = 8 #ignor predictions composed of "threshold_obj" pixels or less
    labeled,n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labeled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result


def predict(img_path, model):
    c_img = imread(img_path)
    img = np.expand_dims(c_img, 0)/255.0
    if IMG_SCALING is not None:
        img = img[:, ::IMG_SCALING[0], ::IMG_SCALING[1]]
    return img, model.predict(img) 


def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_run_length_encoded_predictions(y_pred, img_name):
    list_dict = []
    masks = split_mask(y_pred)
    if len(masks) == 0:
        list_dict.append({"ImageId": img_name, "EncodedPixels": np.nan})
    for mask in masks:
        list_dict.append({"ImageId": img_name, "EncodedPixels": rle_encode(mask)})
    return list_dict


def predict_and_decode(test_img_names):
    list_dict = []
    for img_name in test_img_names:
        _ , pred = predict(img_name)
        rle_pred = get_run_length_encoded_predictions(pred[0], img_name)
        list_dict += rle_pred
    return pd.DataFrame(list_dict, columns=["ImageId", "EncodedPixels"])    


def get_data(dir):
    masks = pd.read_csv(dir)

    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    images_df = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    images_df['has_ships'] = images_df['ships'].map(lambda x: True if x > 0 else False)
    masks.drop(['ships'], axis=1, inplace=True)

    images_with_ships_count = images_df['has_ships'].value_counts()[1]
    images_withot_ships_count = images_df['has_ships'].value_counts()[0]
    sample_to_remove_size = images_withot_ships_count - round(images_with_ships_count * 0.3)
    sample_to_remove = images_df.loc[images_df['has_ships'] == False].sample(sample_to_remove_size)
    images_df = images_df.drop(sample_to_remove.index)

    train_ids, valid_ids = train_test_split(images_df, test_size = 0.02, stratify = images_df['ships'])

    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    return train_df, valid_df