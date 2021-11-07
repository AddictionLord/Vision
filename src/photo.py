import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import measure




# Points for image cropping
TOP_LEFT = (630, 100)
BOT_RIGHT = (1050, 1202)

# THis is substracted from mean value (for thresholding)
# see fcn computeThreshold for more
THRESH_RANGE = 30




class Photo:
    def __init__(self, name: str, photo):
        self.name = name
        # Converts image to GRAY format, only one channel is maintained
        self.photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)


    # ---------------------------------------------------------------------------------------
    # Magic method to display name of the photos in arrays
    def __repr__(self):

        return f"{self.name}"


    # ---------------------------------------------------------------------------------------
    # Image is saved to given directory, path is relative to script src directory
    # Note that image is NOT SAVED when DIRECTORY DOES NOT EXISTS
    def save(self, directory = '../data/processed/'):

        print("Saved to: ", directory)
        cv.imwrite(directory + str(self.name), self.photo)


    # ---------------------------------------------------------------------------------------
    # Image is shown when method is called, press any key to close
    # the image and continue to run the script
    def show(self):

        cv.imshow(self.name, self.photo)
        cv.waitKey(0)
        cv.destroyAllWindows()  


    # ---------------------------------------------------------------------------------------
    # Crops the image, takes top_left and bottom_right corner as arguments,
    # crops is made directly, original image is lost
    def cropImage(self, top_left = TOP_LEFT, bot_right = BOT_RIGHT):

        self.photo = self.photo[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]


    # ---------------------------------------------------------------------------------------
    # Returns mask with edges of top part of the image
    def makeMaskWithDetectedEgdes(self, srcImg):

        # Make copy of original image
        image = srcImg.copy()
        # Crop the image (only top part remains)
        image = image[0:250, 0:image.shape[1]]

        #Detect edges and return mask
        return cv.Canny(image, 120, 2)


    # ---------------------------------------------------------------------------------------
    # Computes mean value of pixels from rectangle placed in the light part of the bottle
    # the photo NEEDS TO BE CROPPED alredy by Photo.cropImage() for this to work!
    def _computeMeanFromCrop(self):

        rectangle = self.photo.copy()[75:375, 100:300]
        return np.mean(rectangle)


    # ---------------------------------------------------------------------------------------
    # Computes threshold of self.photo into self.photo - original image is lost
    # thresh value is computed in fcn, see computeMeanFromCrop()
    def computeThreshold(self, maxval = 255):

        mean = self._computeMeanFromCrop()
        _, self.photo = cv.threshold(self.photo, mean - THRESH_RANGE, maxval, cv.THRESH_BINARY)


    # ---------------------------------------------------------------------------------------
    # Takes self.photo, finds all contours with 10k+ pixels and draws it to the mask
    # which is assigned to self.photo (original image is lost)
    def findLargestContrours(self):

        contours, hierarchy = cv.findContours(self.photo, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(self.photo.shape, np.uint8)
        
        for cnt in contours:
            if 10000<cv.contourArea(cnt):
                cv.drawContours(mask,[cnt], 0, 255, -1)

        self.photo = mask


    # ---------------------------------------------------------------------------------------
    # Pass edges from fcn makeMaskWithDetectedEgdes() and srcImg
    # all pixels with value above 100 will be added to copy of 
    # srcImg which is returned
    def addEdgesToImage(self, edges, srcImg):

        toImg = srcImg.copy()

        rows = edges.shape[0]
        cols = edges.shape[1]
        black = 0
        white = 255

        for row in range(0, rows):
            for col in range(0, cols):
                if edges[row][col] >= 100:
                    toImg[row][col] = white

        return toImg


    # ---------------------------------------------------------------------------------------
    # returns detected edges from the self.photo
    def detectEdges(self):

        return cv.Canny(self.photo, 120, 2)


    # ---------------------------------------------------------------------------------------
    # Takes mask (whatever it is), detects top white point (255) and 
    # bottom white point (255), connects this two points and draw vertical line
    # on the left side of the photo (represents height of empty part - without liquid)
    def findSurface(self):

        for row in range(0, len(self.photo)):

            for column in range(0, len(self.photo[0])):

                if self.photo[row][column] == 255:
                    up = (column, row)
                    break

        for row in range(len(self.photo)-1, 0, -1):

            for column in range(0, len(self.photo[0])):

                if self.photo[row][column] == 255:
                    down = (column, row)
                    break

        delka_cele_flasky_v_milimetrech = 220 # UPRAVTE
        pocet_pixelu_cele_flasky = 990
        pixely_na_milimetry = delka_cele_flasky_v_milimetrech/pocet_pixelu_cele_flasky
        delka_horni_pulky_flasky_v_pixlech =  up[1] - down[1]
        delka_horni_pulky_flasky_v_milimetrech = delka_horni_pulky_flasky_v_pixlech*pixely_na_milimetry
        hladina_v_milimetrech = delka_cele_flasky_v_milimetrech - delka_horni_pulky_flasky_v_milimetrech
        
        print('hladina_v_milimetrech:', hladina_v_milimetrech)

        self.photo = cv.line(self.photo, (20, up[1]), (20, down[1]), (255, 255, 255), 1)
        self.photo = cv.line(self.photo, up, down, (255, 255, 255), 1)




# ---------------------------------------------------------------------------------------
# MAIN - Only for testing/debugging purposes
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Load the image from directory
    img = cv.imread("../data/flaska95.png")

    # Create instance of class Photo
    p = Photo("bottle", img)

    # First crop the photo
    p.cropImage()

    # Detect edges from cropped photo
    edges = p.makeMaskWithDetectedEgdes(p.photo)

    # Threshold the photo
    p.computeThreshold()

    # Find and keep only the largest contour
    p.findLargestContrours()

    # Add edges to largest contour to have mask of whole bottle
    p.photo = p.addEdgesToImage(edges, p.photo)

    # Detect edges on the mask to have only outline of top part of the bottle
    p.photo = p.detectEdges()

    # Find top and bottom white point
    # (cover and surface)
    p.findSurface()

    # Show the processed image
    p.show()

    # Close all windows in the end of the script
    cv.destroyAllWindows()

