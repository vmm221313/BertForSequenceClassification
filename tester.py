import pandas as po
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import format_time
from utils import flat_accuracy


def test(test_dataloader, model, embedding_dim, num_output_classes, device):
    print("")
    print('Testing...')

    model.eval()

    predictions_df = po.DataFrame(columns = ['Prediction', 'Actual Value'])
    for step, batch in enumerate(tqdm(test_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():      
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            raw_predictions = nn.Linear(embedding_dim, num_output_classes)(outputs[1]) 
            prediction = nn.Softmax(dim = 1)(raw_predictions).argmax().item()
            target = b_labels.item()

        row = {'Prediction': prediction,
               'Actual Value': target
              }
        predictions_df = predictions_df.append(row, ignore_index = True)

    predictions_df = predictions_df.astype(int)
    print(classification_report(predictions_df['Prediction'], predictions_df['Actual Value']))
    
    return predictions_df
