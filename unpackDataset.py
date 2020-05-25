# import necessary packages
import os
import cv2

# lookup tables for image ids are provided in .txt files
# the .txt files that should be referenced are listed below
family_txt_files = ["images_family_test.txt", "images_family_trainval.txt",
    "images_family_val.txt", "images_family_train.txt"]
manufacturer_txt_files = ["images_manufacturer_test.txt",
    "images_manufacturer_trainval.txt", "images_manufacturer_val.txt",
    "images_manufacturer_train.txt"]

PATH = "FGVC-AIRCRAFT DATASET/"
IMAGE_PATH = "FGVC-AIRCRAFT DATASET/images/"
SAVE_PATH = "data/"

# create directory for processed dataset
os.mkdir(SAVE_PATH)

# create lookup table for manufacturers
id_to_manufacturer = dict()

for sub_path in manufacturer_txt_files:
    with open(PATH + sub_path) as file:
        for line in file:
            # strip end and split on whitespace
            data = line.rstrip("\n").split()
            image_id = data[0]
            manufacturer_name = "-".join(str(x) for x in data[1:])
            id_to_manufacturer[image_id] = manufacturer_name

# list of image classes that already have directories
dir_exists = list()

# iterate through .txt files
for sub_path in family_txt_files:
    with open(PATH + sub_path) as file:
        # each line contains an image id and a model (e.g. 1025794 Boeing 707)
        for line in file:
            # strip end and split on whitespace
            data = line.rstrip("\n").split()
            image_id = data[0]
            family_name = "-".join(str(x) for x in data[1:])

            # determine class label
            class_label = id_to_manufacturer[image_id] + "_" + family_name

            # if this is the first image of its class, create a directory
            if class_label not in dir_exists:
                print("[INFO] Found", family_name)
                os.mkdir("data/" + class_label.replace("/", ""))
                dir_exists.append(class_label)

            # open image and crop to remove metadata
            image = cv2.imread(IMAGE_PATH + image_id + ".jpg")
            height, width, channels = image.shape
            image = image[0: height - 20, 0:width]

            # save image to class directory
            cv2.imwrite(SAVE_PATH + class_label + "/" + image_id + ".jpg",
                image)
