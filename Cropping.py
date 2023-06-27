import os
import cv2
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

crop_height = 512
crop_width = 512
stride = 256
overlap = int(crop_width - stride)
img_predict_dir = 'Path'
img_result_dir = 'Path'

if not os.path.exists(img_result_dir):
    os.makedirs(img_result_dir)

num = 1
print(img_result_dir)


def fuse(output_image, row, column, overlap, counter, temp1, temp2, res_row, res_column):
    if counter % column == 0:
        output_image = output_image[:, -(res_column + overlap):]
    if counter > (row - 1) * column:
        output_image = output_image[-(res_row + overlap):, :]

    if counter % column == 1:
        temp1 = output_image
    else:
        temp1_1 = temp1[:, 0:-overlap, :]
        temp1_2 = temp1[:, -overlap:, :]
        temp1_3 = output_image[:, 0:overlap, :]
        temp1_4 = output_image[:, overlap:, :]
        temp1_fuse = 0.5 * temp1_2 + 0.5 * temp1_3
        temp1 = np.concatenate((temp1_1, temp1_fuse, temp1_4), axis=1)
        if int(counter % column) == 0:
            if counter == column:
                temp2 = temp1
            else:
                temp2_1 = temp2[0:-overlap, :, :]
                temp2_2 = temp2[-overlap:, :, :]
                temp2_3 = temp1[0:overlap, :, :]
                temp2_4 = temp1[overlap:, :, :]
                temp2_fuse = 0.5 * temp2_2 + 0.5 * temp2_3
                temp2 = np.concatenate((temp2_1, temp2_fuse, temp2_4), axis=0)

    return temp1, temp2


for filename in os.listdir(img_predict_dir):
    img_name = os.path.join(img_predict_dir, filename)
    print(img_name)
    loaded_image = Image.open(img_name)
    img_w, img_h = loaded_image.size
    row = int((img_h - crop_height) / stride + 1)
    column = int((img_w - crop_width) / stride + 1)
    res_row = img_h - (crop_height + stride * (row - 1))
    res_column = img_w - (crop_width + stride * (column - 1))
    if (img_h - crop_height) % stride > 0:
        row += 1
    if (img_w - crop_width) % stride > 0:
        column += 1
    counter = 1

    for i in range(row):
        for j in range(column):
            if i == row - 1:
                H_start = img_h - crop_height
                H_end = img_h
            else:
                H_start = i * stride
                H_end = H_start + crop_height
            if j == column - 1:
                W_start = img_w - crop_width
                W_end = img_w
            else:
                W_start = j * stride
                W_end = W_start + crop_width

            img_chip = loaded_image.crop((W_start, H_start, W_end, H_end))
            img_chip.save(os.path.join(img_result_dir, f'{num:04}.tif'))
            num += 1