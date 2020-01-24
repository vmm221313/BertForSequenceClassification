import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data(df, batch_size):
    input_ids = []
    for sentence in df['text']:
        encoded_sent = tokenizer.encode(sentence, max_length = 120, pad_to_max_length=True)
        input_ids.append(encoded_sent)
    
    attention_masks = []
    for sentence in input_ids:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)
        
    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)
    train_labels = torch.tensor(df['label'].tolist())
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    
    return train_dataloader
