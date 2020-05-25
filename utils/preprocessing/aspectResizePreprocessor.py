# import the necessary packages
import imutils
import cv2

class AspectResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store resizing parameters
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # get image dimensions and initialize sizing deltas
        (h, w) = image.shape[:2]
        deltas = [0, 0]  # dW, dH

        # if height is greater than width, resize along width
        if h > w:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            deltas[0] = int((image.shape[0] - self.height) // 2)

        # else, resize along the height
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            deltas[1] = int((image.shape[1] - self.width) // 2)

        # crop and resize image to target dimensions
        (h, w) = image.shape[:2]
        image = image[deltas[0]:h - deltas[0], deltas[1]:w - deltas[1]]
        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)
