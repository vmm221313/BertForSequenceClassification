import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from trainer import train
from tester import test
from load_model import load_model
from prepare_data import get_dataloader
from preprocess import load_and_preprocess_df

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

num_epochs = 5
batch_size = 32
embedding_dim = 768
num_output_classes = 2
model_type = 'bert-base-uncased'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device {}'.format(device))

train_df, test_df = load_and_preprocess_df()

train_df['label'].value_counts()

tokenizer, model = load_model(model_type)

train_dataloader = get_dataloader(train_df, batch_size, tokenizer, device)

loss_function = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

model.cuda()

model = train(train_dataloader, num_epochs, model, loss_function, optimizer, scheduler, embedding_dim, num_output_classes, device)

test_batch_size = 1
test_dataloader = get_dataloader(test_df, test_batch_size, tokenizer, device)

predictions_df = test(test_dataloader, model, embedding_dim, num_output_classes, device)
