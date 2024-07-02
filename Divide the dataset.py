import os
import random
import shutil
import math

def moveFile(train_img_Dir, train_mask_Dir, test_img_Dir, test_mask_Dir, val_img_Dir, val_mask_Dir):
    img_pathDir = os.listdir(train_img_Dir)
    random.shuffle(img_pathDir)  # Shuffle the list of image files

    total_files = len(img_pathDir)
    test_rate = 0.7
    val_rate = 0.3

    test_picknumber = math.floor(total_files * test_rate)
    val_picknumber = math.floor(total_files * val_rate)

    test_files = img_pathDir[:test_picknumber]
    val_files = img_pathDir[test_picknumber:test_picknumber + val_picknumber]

    move_files(test_files, train_img_Dir, test_img_Dir)
    move_files(test_files, train_mask_Dir, test_mask_Dir)
    move_files(val_files, train_img_Dir, val_img_Dir)
    move_files(val_files, train_mask_Dir, val_mask_Dir)

def move_files(files, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in files:
        name, ext = os.path.splitext(filename)
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)

        # Handle the case where the destination file already exists
        if os.path.exists(destination_file):
            print(f"Warning: Skipping file '{filename}' as it already exists in the destination directory.")
            continue

        shutil.move(source_file, destination_file)

if __name__ == '__main__':
    train_img_Dir = './dataset/img/' #The original unpartitioned dataset
    train_mask_Dir = './dataset/mask/' #The original unpartitioned dataset
    test_img_Dir = './dataset/img/'
    test_mask_Dir = './dataset/mask/'
    val_img_Dir = './dataset/img/'
    val_mask_Dir = './dataset/mask/'

    moveFile(train_img_Dir, train_mask_Dir, test_img_Dir, test_mask_Dir, val_img_Dir, val_mask_Dir)




