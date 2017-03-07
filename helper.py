import errno
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.stats import bernoulli


def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def resize(image, new_dim):
    return cv2.resize(image, new_dim) 

def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return cv2.flip(image,1), -1 * steering_angle
    else:
        return image, steering_angle

def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64)):
    image = crop(image, top_crop_percent, bottom_crop_percent)
    image, steering_angle = random_flip(image, steering_angle)
    image = resize(image, resize_dim)
    return image, steering_angle

def get_next_image_files(batch_size=64):
    data = pd.read_csv('./data/driving_log.csv')
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + 0.25
            image_files_and_angles.append((img, angle))
        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - 0.25
            image_files_and_angles.append((img, angle))
    return image_files_and_angles

def generate_next_batch(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread('./data/'+ img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)
        yield np.array(X_batch), np.array(y_batch)

    