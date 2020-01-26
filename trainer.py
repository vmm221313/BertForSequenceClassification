import time
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import format_time
from utils import flat_accuracy
from saving_and_loading import save_bert_model


def train(train_dataloader, num_epochs, model, loss_function, optimizer, scheduler, embedding_dim, num_output_classes, device):
    losses = []
    for i in range(num_epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(i + 1, num_epochs))
        print('Training...')

        t0 = time.time()
        
        total_loss = 0

        model.train()

        for step, batch in enumerate(tqdm(train_dataloader)):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        
            
            outputs = model(input_ids = b_input_ids, attention_mask = b_input_mask)
            #outputs = (outputs[0].to(device), outputs[1].to(device))
            predictions = outputs[0] ##
            predictions = predictions.to(device) ##

            #dropout = nn.Dropout(p = 0.2).to(device)

            #reg_out = dropout(outputs[1])

            #linear = nn.Linear(embedding_dim, num_output_classes).to(device)
            #predictions = linear(reg_out)    

            targets = b_labels
            targets = targets.to(device)
            predictions = predictions.to(device)
            loss = loss_function(predictions, targets)
                            
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)            
        losses.append(avg_train_loss)
        
        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epoch took: {:}".format(format_time(time.time() - t0)))
        print("")
        
    save_bert_model(model, len(train_dataloader), i+1)
    
    print("")
    print("Training complete!")
    
    return model
