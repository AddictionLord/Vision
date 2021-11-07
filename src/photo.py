import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import measure


TOP_LEFT = (630, 100)
BOT_RIGHT = (1050, 1202)
THRESH_RANGE = 30




class Photo:
    def __init__(self, name: str, photo):
        self.name = name
        self.photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        self.surface_height = None
        self.original = None

        print(self.photo.shape)

        # List of 4 values (for croping the bottle)
        self.boundaries = list()


    def __repr__(self):

        return f"{self.name}"


    def save(self, directory = '../data/processed/'):

        print("Saved to: ", directory)
        cv.imwrite(directory + str(self.name), self.photo)

    
    def show(self):

        cv.imshow(self.name, self.photo)
        cv.waitKey(0)
        cv.destroyAllWindows()  

    
    def computeThreshold(self, boundary, maxval = 255):

        _, self.photo = cv.threshold(self.photo, boundary, maxval, cv.THRESH_BINARY)


    def computeAdaptiveThreshold(self):

        self.photo = cv.adaptiveThreshold(self.photo, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 0)


    def detectEdges(self):

        edges = cv.Canny(self.photo, 120, 2)
        return edges


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


    def drawRectangle(self, start_point, end_point):

        self.photo = cv.rectangle(self.photo, start_point, end_point, (255, 0, 0), 1)


    def cropImage(self, top_left = TOP_LEFT, bot_right = BOT_RIGHT):

        self.photo = self.photo[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
        self.original = self.photo.copy()


    # Computes mean value of pixels from rectangle placed in the light part of the bottle
    # the photo NEEDS TO BE CROPPED alredy by Photo.cropImage() for this to work!
    def computeMeanFromCrop(self):

        rectangle = self.photo[75:375, 100:300]
        return np.mean(rectangle)


    def  smartThreshold(self):

        mean = self.computeMeanFromCrop()
        img = self.photo.copy()
        
        for row in range(0, len(self.photo)):
            for column in range(0, len(self.photo[0])):

                if self.photo[row][column] > mean - THRESH_RANGE:
                    img[row][column] = 255
                else: 
                    img[row][column] = 0

        self.photo = img


    def blur(self):

        self.photo = cv.GaussianBlur(self.photo, (5, 5), 0)


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


    def makeContours(self):

        contours, hierarchy = cv.findContours(self.photo, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # self.photo = cv.drawContours(self.photo, contours, -1, (0,255,0), 3)
        mask = np.zeros(self.photo.shape,np.uint8)
        
        for cnt in contours:
            if 10000<cv.contourArea(cnt):
                print(cv.contourArea(cnt))
                cv.drawContours(self.photo,[cnt],0,255,1)
                cv.drawContours(mask,[cnt],0,255,-1)

        self.photo = mask


    def addEdges(self, edges, toImg):

        rows = edges.shape[0]
        cols = edges.shape[1]
        black = 0
        white = 255

        for row in range(0, rows):
            for col in range(0, cols):
                if edges[row][col] >= 100:
                    toImg[row][col] = white

        return toImg


    def processEdges(self):

        image = self.photo.copy()
        image = image[0:250, 0:image.shape[1]]
        edges = cv.Canny(image, 120, 2)

        return edges



if __name__ == "__main__":

    img = cv.imread("../data/flaska95.png")
    p = Photo("bottle", img)

    # First crop the photo
    p.cropImage()
    edges = p.processEdges()
    p.smartThreshold()
    p.makeContours()
    p.addEdges(edges, p.photo)


    # p.pickRegion()

    # p.findSurface()

    # p.blur()


    # p.drawRectangle((100,75), (300, 375))

    # p.computeThreshold(120)
    # p.computeAdaptiveThreshold()

    # Surface visible
    p.photo = p.detectEdges()

    # Surface visible
    # p.detectCorners(2, 31)
    # p.detectCorners(20, 13)

    # p.drawRectangle(TOP_LEFT, BOT_RIGHT)

    # histogram = cv.calcHist([p.photo],[0],None,[256],[0,256])
    # plt.plot(histogram)
    # plt.show()

    # p.photo = cv.normalize(p.photo, p.photo, 0, 255, cv.NORM_MINMAX)

    # cv.imshow("Original", p.original)
    # print("Final")
    p.show()
    cv.destroyAllWindows()

