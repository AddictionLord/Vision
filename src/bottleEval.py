import numpy as np
import cv2 as cv
import os
import re
import csv
import eye
import photo
#---------------------------------
#CONSTANTS
CUR_PATH = os.path.dirname(__file__)
CUR_PATH_ONE_UP, _ = os.path.split(CUR_PATH)
PATH_DATA = os.path.join(CUR_PATH_ONE_UP,'data')
PATH_DATA_EVALUATED = os.path.join(PATH_DATA,'evaluated')

#BOTTLE_HEIGHT = 195     #[mm]
SURF_LVL = 100           #[mm]
SURF_LVL_TOLERANCE = 10  #[mm]
SURF_LVL_LOWER_BOUND = SURF_LVL - SURF_LVL_TOLERANCE
SURF_LVL_UPPER_BOUND = SURF_LVL + SURF_LVL_TOLERANCE
#---------------------------------
#PROCESS PHOTOS
#e = eye.Eye(PATH_DATA)
#e.load_dataset()

#LOAD PROCESSED PHOTOS
PATH_PROCESSED_IMAGES = os.path.join(PATH_DATA,'processed')

imgProcList = list()
for filee in os.listdir(PATH_PROCESSED_IMAGES):
    png = re.search("png$", filee)

    if png:
        PATH_IMG = os.path.join(PATH_PROCESSED_IMAGES,filee)
        img = cv.imread(PATH_IMG)

        temp = photo.Photo(filee, img)
        imgProcList.append(temp)

#EVALUATE PHOTOS
imgEvalList = list()
imgEvalCounter = [0,0]  #[NumFailed,NumPassed]
for img in imgProcList:
    img.findSurface()
    surf_height = np.round(img.surface_height_in_mm,2)

    if SURF_LVL_LOWER_BOUND <= surf_height and surf_height <= SURF_LVL_UPPER_BOUND:
        imgEvalList.append((img.name,surf_height,"passed"))
        img.drawLimits("passed")
        imgEvalCounter[1] += 1
    else:
        imgEvalList.append((img.name,surf_height,"failed"))
        img.drawLimits("failed")
        imgEvalCounter[0] += 1

    img.save(PATH_DATA_EVALUATED)
print(f"Evaluated Images saved to '{PATH_DATA_EVALUATED}'.")

#PRINT STATS
for item in imgEvalList:
    print(item)
print("[NumFailed,NumPassed]:", imgEvalCounter)

#SAVE RESULTS
output_file = os.path.join(PATH_DATA_EVALUATED,'Bottle_Evals.csv')
with open(output_file,'w', newline='') as new_file:
    csv_writer = csv.writer(new_file, delimiter=';')
    for img in imgEvalList:
        csv_writer.writerow(img)

    print(f"Evaluated data saved to '{output_file}'.")