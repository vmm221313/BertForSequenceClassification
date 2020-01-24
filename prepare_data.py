import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(df, batch_size, tokenizer):
    sentences = df['text'].tolist()
    labels = df['label'].tolist()
    
    #max_len = max([len(sent.split(' ')) for sent in sentences])
    #print(max_len)
    
    max_len = 100
    
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, max_length = max_len, pad_to_max_length=True)
        input_ids.append(encoded_sent)
        
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
        
    assert len(attention_masks) == len(input_ids)
    
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    data = TensorDataset(inputs, masks, labels)
    dataloader = DataLoader(data, batch_size = batch_size)
    
    return dataloader
