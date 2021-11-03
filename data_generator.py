''' 
data_generator.py: This file creates custom datasets for our model.
'''
__author__ = "A.Utkarsh(ayushutkarsh@gmail.com) and M.Gupta(mgmayank18@gmail.com)"


from torch.utils.data import Dataset
import os

class FingerPrintDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.samples = []
        self.transform =transform
        self.root_dir = root_dir
        self.__init__dataset()
        self.label_dict={}
    def __init__dataset(self):
        counter=1
        for folders in os.listdir(self.root_dir):
            folder_path=os.path.join(self.root_dir,folders)
            for image_name in os.listdir(folders):
                img_path = os.path.join(folder_path,image_name)
                label = image_name
                if label not in self.label_dict:
                    self.label_dict['label']=counter
                    counter +=1
                self.samples.append((img_path,self.label_dict['label']))   
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.samples[idx][0]
        label = self.samples[idx][1]
        image = io.imread(img_path)
        sample = {'image': image, 'label': label}
        return sample
