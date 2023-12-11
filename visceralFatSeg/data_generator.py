import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_img(img_dir, img_list,imageHeight,imageWidth,is_mask):
    images = []
    for i, image_name in enumerate(img_list):   
        if (image_name.split('.')[2] == 'gz'):
            image1  = nib.load(image_name).get_fdata()
            image1  = np.squeeze(image1)
            imDim   = image1.shape
            image   = image1 
            if imDim[0] != imageHeight or imDim[1] != imageWidth:
            # 1,  resample the image
                image = resize(image1, (imageHeight, imageWidth), mode='constant', preserve_range=True)
            # 2, normalize the image to uint8 [0,255]
            if is_mask==0:
                image = 255 * (image1 - image1.min()) / (image1.max() - image1.min() + 0.001) # normalize the image to 0~255          
            images.append(image)
    images = np.array(images)
    return(images)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size,imageHeight,imageWidth):

    L = len(img_list)

    while True:

        batch_start = 0
        batch_end   = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit],imageHeight,imageWidth,0)    # 0: image
            Y = load_img(mask_dir, mask_list[batch_start:limit],imageHeight,imageWidth,1)  # 1: mask

            yield (X,Y)      

            batch_start += batch_size   
            batch_end += batch_size


