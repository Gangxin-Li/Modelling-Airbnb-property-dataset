import cv2
from os import walk
import os
import pandas as pd
def download_images():
    return 

def resize_images(address,target_address):
    if not os.path.exists(target_address):
        os.makedirs(target_address)
        print("Create default address:",target_address)
    # shape = []
    for (dirpath, dirnames, filenames) in walk(address):
        for file in filenames:
            # print(dirpath,dirnames,file)
            img = cv2.imread(dirpath+'/'+file)
            img = cv2.resize(img,(540,720),interpolation = cv2.INTER_AREA)
            # print(target_address+'/'+dirnames[0]+'/'+file)
            path = target_address+'/'+dirnames[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("Create folder: ",path)
            cv2.imwrite(path+'/'+file, img)
    #         shape.append(img.shape)
    # table = pd.DataFrame(shape,columns=['height','weighth','channels'])
    # print(table)
    # # print(shape)
    # print(table.info())
    # print(table.describe())

    return


if __name__ == "__main__":
    resize_images('./airbnb-property-listings/images','./processed_images')
    pass