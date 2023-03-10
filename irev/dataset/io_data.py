'''
    Script to define Dataset Class, charge dataset, split in train, test and val and save splited dataset
'''

import pandas as pd
from dataset.split import user_based_split
from dataset.preprocess import *
import os

class RawDataset:
    
    '''
        Dataset Class: Split dataset in train, test and val datas.
        Saves splited datas in datasets folder
        Loads splited datas to dataframes
    '''
    
    def __init__(self, type, path, root_path, split_size : list) -> None:
        
        self.type = type
        self.data_path = path
        self.root_path = root_path
        self.split_size = split_size
        self.columns = ['user_id', 'item_id', 'ratings', 'reviews', 'timestamp']
        
        if type == 'yelp':
            self.data = pd.read_csv(path)
            self.data = self.data[self.columns]
            
        else:
            self.data = pd.read_json(path, lines=True)
            self.data = self.data[self.columns]
        
        
    def text_preprocess(self):
        pass
        
        
    def split_dataset(self):
        
        train, test = user_based_split(self.data, self.split_size[0])
        self.train = pd.DataFrame(train)
        self.test = pd.DataFrame(test)
            
        val = user_based_split(self.test, self.split_size[1])
        self.val = pd.DataFrame(val)
        
            
    def save_splited_datasets(self):
        
        if os.path.exists(self.root_path):
            train_path = self.root_path + "splited_datasets/train" + f"_{self.type}"
            self.train.to_csv(train_path, index=False)
            
            test_path = self.root_path + "splited_datasets/test" + f"_{self.type}"
            self.test.to_csv(test_path, index=False)
            
            val_path = self.root_path + "splited_datasets/val" + f"_{self.type}"
            self.val.to_csv(val_path, index=False)
        else:
            assert(f"Could not find: {self.root_path}")
            
            

class LoadDataset():
    
    def __init__(self, path) -> None:
        self.path = path
        
    def load_splited_datas(self):
        self.train = pd.read_csv(path + "train")
        self.test = pd.read_csv(path + "test")
        self.val = pd.read_csv(path + "val")
        



if __name__ == "__main__":
    '''
        Carregar datasets
        Preprocessar os textos do dataset antes de salvar
        Separar entre treino, teste e validação
        Salvar split
        Carregar split
    '''
    
    type = "amazon"
    path = "path to dataset"
    root_path = "splited_dataset/"
    dataset = RawDataset(type, path, root_path, split_size=[0.8, 0.5])
    
    dataset.text_preprocess()
    
    dataset.split_dataset()
    
    dataset.save_splited_datasets()