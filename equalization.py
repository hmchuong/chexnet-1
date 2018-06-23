import sys
import os
import cv2
import numpy as np
from multiprocessing import Pool
from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

def equalization(image_path, output_path):
    try:
        pil_image = Image.open(image_path).convert('RGB')
        bse_image = np.array(pil_image)#cv2.imread(image_path, 1)#cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        bse_image = bse_image[:, :, ::-1].copy()
        # equ = cv2.equalizeHist(bse_image)
        # cv2.imwrite(output_path,equ)
        #-----Converting image to LAB Color model-----------------------------------
        lab= cv2.cvtColor(bse_image, cv2.COLOR_BGR2LAB)

        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))

        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path,final)
        print("Equalized image {}".format(image_path))
    except:
        print("Cannot equalize image {}".format(image_path))
        print("Unexpected error:", sys.exc_info()[0])


def main(input_folder, output_folder):
    pool = Pool()
    for dirName, subdirList, fileList in os.walk(input_folder):
        i = 0
        total = len(fileList)
        for filename in fileList:
            image_path = os.path.join(dirName,filename)
            output_path = os.path.join(output_folder, filename)
            pool.apply_async(equalization, args=(image_path, output_path, ))
            i += 1
            # sys.stdout.write("\rSteps: {}/{}".format(i,total))
            # sys.stdout.flush()
    pool.close()
    pool.join()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
