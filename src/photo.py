import cv2 as cv
import numpy as np




class Photo:
    def __init__(self, name: str, photo):
        self.name = name
        self.photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        self.surface_height = None


    def __repr__(self):

        return f"{self.name}"

    
    def show(self):

        cv.imshow(self.name, self.photo)
        cv.waitKey(0)

    
    def computeThreshold(self, boundary, maxval = 255):

        _, self.photo = cv.threshold(self.photo, boundary, maxval, cv.THRESH_BINARY)


    def computeAdaptiveThreshold(self):

        self.photo = cv.adaptiveThreshold(self.photo, 180, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 79, 0)


    def detectEdges(self):

        self.photo = cv.Canny(img, 120, 0)


    def detectCorners(self, block_size, aperture):

        corners = cv.cornerHarris(self.photo, block_size, aperture, .05, cv.BORDER_CONSTANT)
        # corners = cv.dilate(corners, None)
        cv.imshow("corners", corners)
        cv.waitKey(0)
        cv.destroyAllWindows()  

        # cv.cvtColor(self.photo, cv.COLOR_GRAY2BGR)
        # print(self.photo.shape)
        # self.photo[corners > 0.01 * corners.max()] = [0, 0, 255]
        # print(corners.shape)







if __name__ == "__main__":

    img = cv.imread("../data/flaska.png")
    p = Photo("bottle", img)
    p.show()

    # p.computeThreshold(120)
    # p.computeAdaptiveThreshold()

    # Surface visible
    p.detectEdges()

    # Surface visible
    # p.detectCorners(2, 31)
    # p.detectCorners(20, 13)

    p.show()
    cv.destroyAllWindows()




