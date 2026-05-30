import torch
import numpy as np
import pandas as pd
import json
import os
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm

class Twibot20(Dataset):
    def __init__(self,root='../data/processed_data/',device='cpu'):
        self.root = root
        self.device = device
        
    def load_labels(self):
        print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        labels=torch.load(self.root+"label.pt").to(self.device)
        print('Finished')
        
        return labels

    def Des_embbeding(self):
        print('Running feature1 embedding')
        path=self.root+"des_tensor.pt"
        des_tensor=torch.load(path).to(self.device)
        print('Finished')
        return des_tensor
    
    def tweets_embedding(self):
        print('Running feature2 embedding')
        path=self.root+"tweets_tensor.pt"
        tweets_tensor=torch.load(path).to(self.device)
        print('Finished')
        return tweets_tensor
    
    def num_prop_preprocess(self):
        print('Processing num_properties...',end='   ')
        path = self.root+'num_properties_tensor.pt'
        num_prop=torch.load(path).to(self.device)
        print('Finished')
        return num_prop
    
    def cat_prop_preprocess(self):
        print('Processing cat_properties...',end='   ')
        path=self.root+'cat_properties_tensor.pt'
        category_properties=torch.load(path).to(self.device)
        print('Finished')
        return category_properties
    
    def Build_Graph(self):
        print('Building graph',end='   ')
        path_edge_index=self.root+'edge_index_all.pt'
        path_edge_type=self.root+'edge_type_all.pt'
        edge_index=torch.load(path_edge_index).to(self.device)
        edge_type=torch.load(path_edge_type).to(self.device)
        print('Finished')
        return edge_index,edge_type

    def Build_Graph2(self):
        print('Building graph',end='   ')
        path_edge_index=self.root+'edge_index.pt'
        path_edge_type=self.root+'edge_type.pt'
        edge_index=torch.load(path_edge_index).to(self.device)
        edge_type=torch.load(path_edge_type).to(self.device)
        print('Finished')
        return edge_index,edge_type
    
    def train_val_test_mask(self):
        train_idx=torch.load(self.root+'train_idx.pt')
        val_idx=torch.load(self.root+'val_idx.pt')
        test_idx=torch.load(self.root+'test_idx.pt')
            
        return train_idx,val_idx,test_idx

    def semantic_embedding_gpt(self):
        print('Processing semantic_embedding...',end='   ')
        path=self.root+'embeddings_iter_gpt.pt'
        semantic_tensor=torch.load(path).to(self.device)
        print('Finished')
        return semantic_tensor
    
    def semantic_embedding_deepseek(self):
        print('Processing semantic_embedding...',end='   ')
        path=self.root+'embeddings_iter_deepseek.pt'
        semantic_tensor=torch.load(path).to(self.device)
        print('Finished')
        return semantic_tensor
    
    def semantic_embedding_llama(self):
        print('Processing semantic_embedding...',end='   ')
        path=self.root+'embeddings_iter_llama.pt'
        semantic_tensor=torch.load(path).to(self.device)
        print('Finished')
        return semantic_tensor
    
    def load_shb_prompt_pretrain(self):
        with open("../data/processed_data/Shb_prompt.json","r", encoding="utf-8") as file:
            promots = json.load(file)
        return list(promots.values())

    def load_stx_prompt_pretrain(self):
        with open("../data/processed_data/Stx_prompt.json","r", encoding="utf-8") as file:
            promots = json.load(file)
        return list(promots.values())

    def get_data(self,need_prompts=True):
        labels=self.load_labels()
        des_tensor=self.Des_embbeding()
        tweets_tensor=self.tweets_embedding()
        num_prop=self.num_prop_preprocess()
        category_prop=self.cat_prop_preprocess()
        edge_index,edge_type=self.Build_Graph()
        train_idx,val_idx,test_idx=self.train_val_test_mask()

        if need_prompts:
            prompts = self.llm_prompts()
            return prompts,des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx
        else:
            return des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx
    
    def get_data2(self,semantic_type):
        labels=self.load_labels()
        des_tensor=self.Des_embbeding()[:11826]
        tweets_tensor=self.tweets_embedding()[:11826]
        num_prop=self.num_prop_preprocess()[:11826]
        category_prop=self.cat_prop_preprocess()[:11826]
        if semantic_type == 'gpt':
            semantic_tensor=self.semantic_embedding_gpt()
        elif semantic_type == 'deepseek':
            semantic_tensor=self.semantic_embedding_deepseek()
        elif semantic_type == 'llama':
            semantic_tensor=self.semantic_embedding_llama()
        edge_index,edge_type=self.Build_Graph2()
        train_idx,val_idx,test_idx=self.train_val_test_mask()

        return des_tensor,tweets_tensor,semantic_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx


class MyDataset(Dataset):
    def __init__(self,data_idx,labels):
        self.data_idx = data_idx
        self.labels = labels
        # self.prompts = '''Given a Tweet user's metadata,text information,topology information,and the fusion information of the three,please evaluate whether it is a human or a bot?\nMetadata:<metadata>[SEP]Text information:<text>[SEP]Topology information:<graph>[SEP]Fusion information:<fusion>'''
        self.prompts = '''Given a Tweet user's metadata,text information,topology information,and the fusion information of the three,please evaluate whether it is a human or a bot?\nMetadata:<metadata>\nText information:<text>\nTopology information:<graph>\nFusion information:<fusion>'''
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data_idx[idx],self.prompts,self.labels[idx]

class MyDataset_without_TRM(Dataset):
    def __init__(self,data_idx,labels):
        self.data_idx = data_idx
        self.labels = labels
        # self.prompts = '''Given a Tweet user's metadata,text information,topology information,and the fusion information of the three,please evaluate whether it is a human or a bot?\nMetadata:<metadata>[SEP]Text information:<text>[SEP]Topology information:<graph>'''
        self.prompts = '''Given a Tweet user's metadata,text information,topology information,and the fusion information of the three,please evaluate whether it is a human or a bot?\nMetadata:<metadata>\nText information:<text>\nTopology information:<graph>'''
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data_idx[idx],self.prompts,self.labels[idx]

class MyDataset_without_graph(Dataset):
    def __init__(self,data_idx,labels):
        self.data_idx = data_idx
        self.labels = labels
        # self.prompts = '''Given a Tweet user's metadata,text information,topology information,and the fusion information of the three,please evaluate whether it is a human or a bot?\nMetadata:<metadata>[SEP]Text information:<text>[SEP]Topology information:<graph>'''
        self.prompts = '''Given a Tweet user's metadata,text information,and the fusion information of the two,please evaluate whether it is a human or a bot?\nMetadata:<metadata>\nText information:<text>\nFusion information:<fusion>'''
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data_idx[idx],self.prompts,self.labels[idx]

class MyDataset_without_metadata(Dataset):
    def __init__(self,data_idx,labels):
        self.data_idx = data_idx
        self.labels = labels
        self.prompts = '''Given a Tweet user's text information,topology information,and the fusion information of the two,please evaluate whether it is a human or a bot?\nText information:<text>\nTopology information:<graph>\nFusion information:<fusion>'''
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data_idx[idx],self.prompts,self.labels[idx]
    
class MyDataset_without_text(Dataset):
    def __init__(self,data_idx,labels):
        self.data_idx = data_idx
        self.labels = labels
        self.prompts = '''Given a Tweet user's metadata,topology information,and the fusion information of the two,please evaluate whether it is a human or a bot?\nMetadata:<metadata>\nTopology information:<graph>\nFusion information:<fusion>'''
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data_idx[idx],self.prompts,self.labels[idx]
