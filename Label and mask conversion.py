import numpy as np
import os
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageOperation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def saveImage(image, path):
        image.save(path)

def label_annotation(img):
    label = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    black_indices = np.where((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    white_indices = np.where((img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255))
    label[black_indices[0], black_indices[1]] = 1  # back=1
    label[white_indices[0], white_indices[1]] = 0  # white=0
    return Image.fromarray(np.uint8(label))

def process_color(color_path, label_path, color_name):
    tmp_color_name = os.path.join(color_path, color_name)
    color_label = ImageOperation.openImage(tmp_color_name)
    color_label.load()
    color_label = np.asarray(color_label)
    label = label_annotation(color_label)

    tmp_label_name = os.path.join(label_path, color_name)

    ImageOperation.saveImage(label, tmp_label_name)

def Color2label(color_path, label_path):
    if os.path.isdir(color_path):
        color_names = os.listdir(color_path)
    else:
        color_names = [color_path]

    Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_color)(color_path, label_path, color_name) for color_name in tqdm(color_names)
    )


def color_annotation(img):
    color = np.ones([img.shape[0], img.shape[1], 3])
    color[img == 0] = [255, 255, 255]  # white
    color[img == 1] = [0, 0, 0]  # black
    return Image.fromarray(np.uint8(color))

def Label2Color(label_path, color_label_path):
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for i in range(len(os.listdir(label_path))):
            label_name = label_names[i]
            tmp_label_name = os.path.join(label_path, label_name)

            futures.append(executor.submit(process_image, tmp_label_name, color_label_path, label_name))

        for future in tqdm(as_completed(futures), total=len(futures)):
            pass

def process_image(tmp_label_name, color_label_path, label_name):
    label = ImageOperation.openImage(tmp_label_name)
    label.load()
    label = np.asarray(label)
    color_label = color_annotation(label)
    tmp_color_label_name = os.path.join(color_label_path, label_name)
    ImageOperation.saveImage(color_label, tmp_color_label_name)
if __name__ == '__main__':
    Label2Color("./mask/", "./label/") # mask to label
    #Color2label("./label/",  "./mask/")  #label to mask

