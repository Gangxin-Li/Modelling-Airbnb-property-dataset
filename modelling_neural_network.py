from PIL import Image
from tabular_data import load_airbnb
import torch
import os
from os import walk
import pandas as pd
class dataset(torch.utils.data.Dataset):
    @classmethod
    def from_folder_csv(self,img_path,labels):
        img = []
        prices = []
        for (dirpath, dirnames, filenames) in walk(img_path):
        
            folder_file = []
            for file in filenames:
                image = Image.open(dirpath+'/'+file)
                folder_file.append(image)
                
           
            ID_check = dirpath.split('/')[-1]
            if ID_check == 'processed_images':
                continue
            # print(len(labels[labels['ID'] == ID_check]['ID'] == ID_check))
            if len(labels[labels['ID']==ID_check]['ID'] == ID_check):
                img.append(folder_file)
                prices.append(labels[labels['ID']==ID_check]['Price_Night'])
                # print("Got one")
            else:
                pass
                # print("drop one")
            
        return img,prices
    def __init__(self, img_path, file):
        super().__init__()

        
        self.img,self.label = self.from_folder_csv(img_path,file)
        

    # Not dependent on index
    def __getitem__(self, index):
        return self.img[index], self.label[index]

    def __len__(self):
        return len(self.img)












if __name__ == "__main__":
    img_path = './processed_images'
    # table,labels = load_airbnb(['ID','Price_Night'])
    table = pd.read_csv('./clean_tabular_data.csv')
    labels = table[['ID','Price_Night']]
    dataset = dataset(img_path,labels)
    # print(dataset[:50])
    # print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    
