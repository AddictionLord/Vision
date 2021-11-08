import cv2 as cv
import re 
import sys
import os

from photo import Photo

CUR_PATH = os.path.dirname(__file__)
CUR_PATH_ONE_UP, _ = os.path.split(CUR_PATH)
PATH_DATA = os.path.join(CUR_PATH_ONE_UP,'data')




class Eye:

    def __init__(self, dataset_directory: str):
        self.directory = dataset_directory
        self.data = list()


    # ---------------------------------------------------------------------------------------
    # Iterates over self.directory files, if the file format is png
    # the file (image) is loaded, instantiated (class Photo) and
    # processed by self.inspect() fcn 
    def load_dataset(self):

        for filee in os.listdir(self.directory):
            png = re.search("png$", filee)

            if png:
                imgPATH = os.path.join(self.directory, filee)
                print("Image:", imgPATH)
                img = cv.imread(imgPATH)

                temp = Photo(filee, img)
                self.inspect(temp)


    # ---------------------------------------------------------------------------------------
    # Argument img is instance of class Photo, this
    # does all the image processing and saves the result
    def inspect(self, img):

        # First crop the photo
        # (original photo is lost)
        img.cropImage()

        # Detect edges from cropped photo (only the top part)
        edges = img.makeMaskWithDetectedEgdes(img.photo)

        # Threshold the photo 
        # (original photo is lost)
        img.computeThreshold()

        # Find and keep the largest contours only 
        # (original photo is lost)
        img.findLargestContrours()

        # Add edges to largest contour to have mask of whole bottle,
        # result is returned from the fcn
        img.photo = img.addEdgesToImage(edges, img.photo)

        # Detect edges on the mask to have only outline of top part of the bottle
        # (original photo is lost)
        img.photo = img.detectEdges()

        # Find top and bottom white point (bottle top and surface)
        # (added to original photo - img.photo)
        img.findSurface()

        # Save the image
        img.save()




# ---------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    e = Eye(PATH_DATA)
    e.load_dataset()
