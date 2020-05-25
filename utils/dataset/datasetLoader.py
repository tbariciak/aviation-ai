# import the necessary packages
import numpy as np
import cv2
import os

class DatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessors
        self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class labels assuming that our
            # path has format: /path/{label1}_{label2}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            label = label.replace("_", " ")

            # if preprocessors are specified
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            # show an update if necessary
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return(np.array(data), np.array(labels))
