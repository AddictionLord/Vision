import cv2 as cv
import re 
import sys
import os

from photo import Photo



'''`
ARCHITECTURE IDEAS:

1. Load whole dataset to memory (self.data), process images after

2. Load images one by one and call self.inspect function in self.load_dataset
'''




class Eye:

    def __init__(self, dataset_directory: str):
        self.directory = dataset_directory
        self.data = list()


    def load_dataset(self):

        for filee in os.listdir(self.directory):
            png = re.search("png$", filee)

            if png:
                print(self.directory + filee)
                img = cv.imread(self.directory + filee)

                temp = Photo(filee, img)
                self.inspect(temp)

                # Not necessary to keep whole dataset in memory,
                # could inspect photoes one by one
                # self.data.append(img)


    def inspect(self, img):

        # img.cropImage()
        # img.smartThreshold()
        # # img.show()

        # img.makeContours()
        # img.findSurface()
        # # cv.imshow("Original", img.original)
        # # img.show()
        # img.save()

        img.cropImage()
        edges = img.processEdges()
        img.smartThreshold()
        img.makeContours()
        img.addEdges(edges, img.photo)

        # Surface visible
        img.photo = img.detectEdges()
        img.findSurface()
        img.save()
        # img.show()






if __name__ == "__main__":

    e = Eye("../data/")
    e.load_dataset()
