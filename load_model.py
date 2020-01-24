import torch
from transformers import *

models = {'bert-base-uncased': (BertModel, BertTokenizer),
          'openai-gpt': (OpenAIGPTModel, OpenAIGPTTokenizer),
          'gpt2': (GPT2Model, GPT2Tokenizer),
          'ctrl': (CTRLModel, CTRLTokenizer),
          'transfo-xl-wt103': (TransfoXLModel, TransfoXLTokenizer),
          'xlnet-base-cased': (XLNetModel, XLNetTokenizer),
          'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer),
          'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
          'roberta-base': (RobertaModel,    RobertaTokenizer),
         }


def load_model(model_type):
    model = models['bert-base-uncased'][0].from_pretrained(model_type, output_hidden_states=False)
    tokenizer = models['bert-base-uncased'][1].from_pretrained(model_type)
    return tokenizer, model
