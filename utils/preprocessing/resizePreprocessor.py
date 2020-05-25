# import the necessary packages
import cv2

class ResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store resizing parameters
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # return resized image
        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)
