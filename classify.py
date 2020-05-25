# import the necessary packages
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import AspectResizePreprocessor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import imutils
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load and preprocess the image
image = cv2.imread(args["image"])
img_copy = image.copy()
image = AspectResizePreprocessor(64, 64).preprocess(image)
image = ImageToArrayPreprocessor().preprocess(image)
image = img_to_array(image.astype("float") / 255.0)
image = np.expand_dims(image, axis=0)

# load the pre-trained network and label binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image
print("[INFO] classifying image...")
preds = model.predict(image)[0]
idx = np.argmax(preds)
label = lb.classes_[idx]

# annotate input image
img_copy = imutils.resize(img_copy, width=400)
label = "{}: {:.2f}%".format(label, preds[idx] * 100)
cv2.putText(img_copy, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    (0, 255, 0), 2)

# show the output image
print("[INFO] classified as {}".format(label))
cv2.imshow("Classification", img_copy)
cv2.waitKey(0)
