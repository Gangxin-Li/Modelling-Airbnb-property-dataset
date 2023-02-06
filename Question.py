import os
from os import walk
img_path = './processed_images'
for (dirpath, dirnames, filenames) in walk(img_path):
    print(type(filenames))
    print(filenames)
    filenames = list(filenames)
    print(filenames[0])
    ID = dirpath.split('/')
 
    # print(ID[-1])
    print(dirpath)
   