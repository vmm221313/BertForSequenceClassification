import os

import torch
import torch.nn as nn

from transformers import BertModel
from transformers import BertConfig


def save_bert_model(model, num_sents, num_epochs):
    #save_dir = 'saved_models/'
    save_dir = 'saved_model_model__num_sents={}__num_epochs={}/'.format(num_sents, num_epochs)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving model to %s" % save_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
        
    print('BertModel succesfully saved at {}'.format(save_dir))


def load_bert_model(load_path, device):
    model = BertModel.from_pretrained(load_path)
    model.to(device)
    
    return model

# def save_bert_model(model, num_sents, num_epochs):
#     save_file = 'saved_models/model__num_sents={}__num_epochs={}'.format(num_sents, num_epochs)
#     config_file = 'saved_configs/model__num_sents={}__num_epochs={}'.format(num_sents, num_epochs)
#     
#     model_to_save = model.module if hasattr(model, 'module') else model
#     torch.save(model_to_save.state_dict(), save_file)
#     
#     with open(config_file, 'w') as model_file:
#         model_file.write(model.config.to_json_string())
#         
#     print('BertModel succesfully saved at {}'.format(save_file))

# def load_bert_model(save_file, config_file):
#     config = BertConfig(config_file)
#     model = BertModel(config)
#     model.load_state_dict(torch.load(save_file)) 
#     
#     return model
