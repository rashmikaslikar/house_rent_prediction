import numpy as np 
import pandas as pd
import yaml
import argparse
import os
from pathlib import Path
import pickle
import sys
sys.path.insert(0,os.getcwd())
import file_io.get_path as getpath
from data.preprocess import *


task2=True
class Dataset_Prediction(object):
    def __init__(self,mode,dataset_path,config_file_path,statistics,log_path):
        """
        Custom PyTorch Dataset to store data samples and their corresponding labels.

        Args:
                
        """
        path_getter = getpath.GetPath()
        dataset_folder = path_getter.get_data_path()
        data_df=pd.read_csv(os.path.join(dataset_folder,dataset_path))[0:20000]
        config_file_path=os.path.join(Path.cwd(),'data',config_file_path)
        print(f'Raw data shape: {data_df.shape}')
        with open(config_file_path, "r") as config:
            config = yaml.safe_load(config)
        config=pd.DataFrame(config)

        "Remove unwanted metrics"
        data_mod=drop_metrics(data_df, config)

        "Remove rows where target metric is 0 or missing"
        data_mod=clean_target(data_mod)

        "Remove outliers from the numerical metrics"
        data_mod=remove_outliers(data_mod)
        print(data_mod.shape)

        print(f'Modified data shape: {data_mod.shape}')

        text_fields=['description','facilities']
        for metric in text_fields:
            data_mod=data_mod.loc[~data_mod[metric].isnull()]

        df_text=data_mod[text_fields]
        data_mod.drop(text_fields,axis=1,inplace=True)
        print(f'Modified data shape: {data_mod.shape}')

        if task2:
            "generate embeddings for text data"
            data_text=process_text(df_text,mode)

        data_num,target=process_num_data(data_mod,statistics,log_path)
        data_cat,df_bool_from_cat=process_cat_data(data_mod) 
        data_bool=process_bool_data(data_mod,df_bool_from_cat)       

        self.targets=target

        if not task2:
            self.inputs=np.concatenate((data_num,data_cat,data_bool),axis=1)
            print(f'input,target:{self.inputs.shape},{self.targets.shape}')
        else:
            self.inputs=np.concatenate((data_num,data_cat,data_bool,data_text),axis=1)        

    def return_feature_length(self):
        """ Function that returns the number of features in the data. """
        feature_length = self.inputs.shape[1]
        return feature_length
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        input_data=self.inputs[idx]
        ground_truth=self.targets[idx]
        sample = {'input_data':input_data.astype('double'),'gt':ground_truth.astype('double')} 
        return sample
    
    def normalize(self):
        pass

    def return_statistics(self):
        pass  

      

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-dataset-path', default='immo_train.csv', type=str, 
                       help='Path to the dataset .csv file')
   parser.add_argument('-config-path', default='config.yaml', type=str, 
                       help='Path to the config file')
   args = parser.parse_args()

   loader=Dataset_Prediction('beta_test',args.dataset_path,args.config_path,None,os.path.join(Path.cwd(),'dataloader'))
    


        

