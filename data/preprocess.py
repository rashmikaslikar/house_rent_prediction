import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from scipy import stats
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch.nn as nn

def drop_metrics(dataframe, config):
    ignore_metrics=config[config['ignore']==True]['metric'].tolist()
    df=dataframe.drop(ignore_metrics, axis=1)
    return df

def clean_target(dataframe):
    df=dataframe.loc[~dataframe['totalRent'].isnull()]
    df=df.loc[df['totalRent']!=0]
    return df

def remove_outliers(df):
    metrics=['serviceCharge', 'livingSpace', 'totalRent']    
    for metric in metrics:
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[metric] > (q1 - 1.5 * iqr)) & (df[metric] < (q3 + 1.5 * iqr))]
    return df

def process_cat_data(df):
    #get DataFrame with categorical (non numeric columns)
    df_cat = df.iloc[:, :-1].select_dtypes(object)
    df.drop(df_cat.columns,axis=1,inplace=True)

    "Metric heatingType"
    other=['oil_heating','heat_pump','combined_heat_and_power_plant','night_storage_heater','wood_pellet_heating','electric_heating','stove_heating','solar_heating']
    df_cat['heatingType'] = df_cat['heatingType'].apply(lambda x: 'other' if x in other else x)

    "Metric condition"
    other=['negotiable','need_of_renovation','ripe_for_demolition']
    df_cat['condition'] = df_cat['condition'].apply(lambda x: 'other' if x in other else x)

    "Metric interiorQual"
    df_cat.loc[ df_cat['interiorQual'] == 'simple','interiorQual'] = 'normal'

    "Metric typeOfFlat - convert to boolean"
    luxury=['maisonette','terraced_flat','penthouse','loft']
    non_luxury=['ground_floor','apartment','roof_storey','raised_ground_floor','other','half_basement']
    df_cat['typeOfFlat'] = df_cat['typeOfFlat'].apply(lambda x: 'luxury' if x in luxury else x)
    df_cat['typeOfFlat'] = df_cat['typeOfFlat'].apply(lambda x: 'non_luxury' if x in non_luxury else x)
    df_cat['typeOfFlat'].fillna('non_luxury',inplace=True)
    df_luxury = pd.get_dummies(pd.DataFrame(df_cat['typeOfFlat']), drop_first=True)
    df_luxury=df_luxury.astype(bool)
    df_cat.drop('typeOfFlat',axis=1,inplace=True)

    "Metric telekomTvOffer convert to boolean"
    df_cat.loc[df_cat['telekomTvOffer'] == 'ON_DEMAND','telekomTvOffer'] = 'NONE'
    df_cat['telekomTvOffer'].fillna('NONE',inplace=True)
    df_Telekom = pd.get_dummies(pd.DataFrame(df_cat['telekomTvOffer']), drop_first=True)
    df_Telekom=df_Telekom.astype(bool)
    df_cat.drop('telekomTvOffer',axis=1,inplace=True)

    "Metric petsAllowed"
    criteria=['yes','negotiable']
    df_cat['petsAllowed'] = df_cat['petsAllowed'].apply(lambda x: 'yes_or_negotiable' if x in criteria else x)
    df_cat['petsAllowed'].fillna('no',inplace=True)
    df_pets = pd.get_dummies(pd.DataFrame(df_cat['petsAllowed']), drop_first=True)
    df_pets=df_pets.astype(bool)
    df_cat.drop('petsAllowed',axis=1,inplace=True)

    df_bool=pd.concat([df_luxury, df_Telekom,df_pets], axis=1)

    for feature in df_cat.select_dtypes(include=['object']).columns:
        df_cat[feature].fillna('unknown',inplace=True) 

    #one hot encoding of categorical metrics
    df_cat =  pd.get_dummies(df_cat,sparse=True)
    data=df_cat.values

    #df_final=pd.concat([df_bool, df_cat, df], axis=1)
    return data,df_bool

def process_num_data(df,statistics,log_path):    
    df_num = df.iloc[:, :-1].select_dtypes(include=['number']) 
    data=df_num.values
    
    if statistics==None:
        statistics = {}
        mean=np.nanmean(data,axis=0)
        minimum=np.nanmin(data,axis=0)
        maximum=np.nanmax(data,axis=0)
        mode=stats.mode(df_num['telekomUploadSpeed'].values,keepdims=True)[0][0]   
        statistics['mean'] = mean
        statistics['mode'] = mode
        statistics['maximum'] = maximum
        statistics['minimum'] = minimum
        dump_dictionary(statistics,log_path)
    else:
        mean=statistics['mean']
        minimum=statistics['minimum']
        maximum=statistics['maximum']
        mode=statistics['mode']

    for n,col in enumerate(df_num.columns):
        if col=='telekomUploadSpeed':
            df[col].fillna(mode,inplace=True)
        else:
            df[col].fillna(mean[n],inplace=True)
    data=normalize_num_data(df)
    return data  

def normalize_num_data(df):
    df_num = df.iloc[:, :-1].select_dtypes(include=['number']) 
    target=df_num['totalRent'].values.reshape(-1,1)
    df_num.drop('totalRent',axis=1,inplace=True)
    data=df_num.values
    norm = MinMaxScaler().fit(data)
    data_norm = norm.transform(data)
    norm=MinMaxScaler().fit(target)
    target_norm = norm.transform(target)
    return data_norm,target_norm

def process_bool_data(df,df_bool_from_cat):
    df_bool = df.iloc[:, :-1].select_dtypes(include=['boolean']) 
    df_final=pd.concat([df_bool, df_bool_from_cat], axis=1).astype(int)
    data_bool=df_final.values
    return data_bool 
    
def process_text(df,mode):
    text_embeddings=[]
    model_name = "bert-base-german-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    #df.fillna('unknown', inplace=True)
    for column in df.columns:
        col_embed=[]
        for i,text in enumerate(df[column]):
            #print(i)
            col_embed.append(get_text_embeddings(text, tokenizer, model))
            reduced_embed=dense(torch.tensor(col_embed))
            #print(np.array(col_embed).shape)
            #col_embeddings.append(average_embedding)
        text_embeddings.append(reduced_embed.detach().cpu().numpy())
    text_embeddings=np.concatenate((text_embeddings),axis=1)
    #np.save(mode+'_embeddings',text_embeddings)
    #text_embeddings.append(dataframe['description'].values.apply(lambda x: get_text_embeddings(x, tokenizer, model)))
    #a=torch.tensor(text_embeddings)
    #torch.cat((a,b.view(-1, 1)), dim=1)
    return text_embeddings

def get_text_embeddings(text, tokenizer, model):
    # Tokenize and convert text to embeddings
    #print(text)
    input_ids = tokenizer.encode(text, return_tensors="pt",truncation=True)
    with torch.no_grad():
        outputs = model(input_ids)

    # The embeddings are contained in the last layer of the model's output
    last_hidden_states = outputs.last_hidden_state

    # You can extract the embedding for each token in the input
    # For a simple example, you can average the embeddings across all tokens
    average_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()

    return average_embedding

def pca(embeddings):
    # Apply PCA for dimensionality reduction
    num_components = 50  # Set the desired number of components
    pca = PCA(n_components=num_components)
    text_embeddings_reduced = pca.fit_transform(embeddings)
    return text_embeddings_reduced

def dense(embeddings):
    # Apply a dense layer for dimensionality reduction
    input_size = embeddings.shape[1]
    output_size = 50  # Set the desired output size
    dense_layer = nn.Linear(input_size, output_size)
    text_embeddings_reduced = dense_layer(embeddings)
    #print(text_embeddings_reduced.shape) 
    return text_embeddings_reduced   


def dump_dictionary(dictionary,log_path):  
        #print(self.log_path)  
        with open(os.path.join(log_path,'parameters.pickle'), 'wb') as fp:
            pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)