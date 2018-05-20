import sys
import os
import cv2
import numpy as np
from multiprocessing import Pool

def equalization(image_path, output_path):
    try:
        bse_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        equ = cv2.equalizeHist(bse_image)
        cv2.imwrite(output_path,equ)
    except:
        print("Cannot equalization image {}".format(image_path))
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
            sys.stdout.write("\rSteps: {}/{}".format(i,total))
            sys.stdout.flush()
    pool.close()
    pool.join()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
