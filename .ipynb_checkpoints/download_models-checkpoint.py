import os
import torch
from transformers import *

MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          #(XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
         ]

for model_class, tokenizer_class, pretrained_weights in MODELS:
    print(pretrained_weights)
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    os.mkdir('HuggingFace_Models/{}'.format(pretrained_weights))
    os.mkdir('HuggingFace_Models/{}/model'.format(pretrained_weights))
    os.mkdir('HuggingFace_Models/{}/tokenizer'.format(pretrained_weights))
    
    tokenizer.save_pretrained('HuggingFace_Models/{}/tokenizer/'.format(pretrained_weights))    
    model.save_pretrained('HuggingFace_Models/{}/model/'.format(pretrained_weights))



# +
    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in BERT_MODEL_CLASSES:
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained(pretrained_weights)

    # Models can return full list of hidden-states & attentions weights at each layer
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_attentions = model(input_ids)[-2:]

    # Models are compatible with Torchscript
    model = model_class.from_pretrained(pretrained_weights, torchscript=True)
    traced_model = torch.jit.trace(model, (input_ids,))

    # Simple serialization for models and tokenizers
    model.save_pretrained('./directory/to/save/')  # save
    model = model_class.from_pretrained('./directory/to/save/')  # re-load
    tokenizer.save_pretrained('./directory/to/save/')  # save
    tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load

    # SOTA examples for GLUE, SQUAD, text generation...
