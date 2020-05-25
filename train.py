# import the necessary packages
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import AspectResizePreprocessor
from utils.dataset import DatasetLoader
from utils.nn import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os

# argument parser for input parameters
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to output label binarizer")
ap.add_argument("-w", "--weights", required=True,
    help="path to weights directory")
args = vars(ap.parse_args())

# images will be resized to 64x64 without changing the aspect ratio
arp = AspectResizePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk and scale pixel intensities to [0, 1]
imagePaths = list(paths.list_images(args["dataset"]))
dl = DatasetLoader(preprocessors=[arp, iap])
(data, labels) = dl.load(imagePaths, verbose=100)
data = data.astype("float") / 255.0
print(labels)

# extract class names from iamge labels
classNames = [str(x) for x in np.unique(labels)]
print(classNames)

# split data 75%/25% for training and testing respectively
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
    random_state=42)

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
print(lb.classes_)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=False, fill_mode="nearest")  # horizontal flip true

# initialize the compiler and model
print("[INFO] compiling model...")
# opt = Adam(lr=1e-3, decay=1e-3 / 100)  # SGD(lr=0.1)
opt = SGD(lr=0.005)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# save the model every time the validation loss improves
fname = os.path.sep.join([args["weights"],
    "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
    save_best_only=True, verbose=1)

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
    epochs=100, callbacks=[checkpoint], verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the multi-label binarizer
print("[INFO] Saving label binarizer...")
file = open(args["labelbin"], "wb")
file.write(pickle.dumps(lb))
file.close()

# evaluate network
print("[INFO] evaluating network...")
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
    target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
