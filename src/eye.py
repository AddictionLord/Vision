import cv2 as cv
import re 
import sys
import os



'''
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

                # [DELETE] - Used just to view dataset for debugging purpose
                cv.imshow(filee, img)
                cv.waitKey(0)

                # Not necessary to keep whole dataset in memory,
                # could inspect photoes one by one
                self.data.append(img)


    def inspect():

        # Process images
        pass








if __name__ == "__main__":

    e = Eye("../data/")
    e.load_dataset()
