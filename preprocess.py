import os
import nltk
import pandas as po


### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")", "@",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1


def load_and_preprocess_df():
    train = po.read_csv('data/task6_train.csv').drop('Unnamed: 0', axis = 1)
    test = po.read_csv('data/task6_test.csv').drop('Unnamed: 0', axis = 1)
    
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    
    stopwords = list(set(nltk.corpus.stopwords.words("english")))

    ### tokenize & remove funny characters
    train["text"] = train["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords)).apply(lambda x: ' '.join(x))
    test["text"] = test["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords)).apply(lambda x: ' '.join(x))
    
    empty_sent_index = []
    for i, sent in enumerate(train['text']):
        if len(sent) == 0:
            empty_sent_index.append(i)

    train = train.drop(empty_sent_index, axis = 0).reset_index(drop = True)
    
    empty_sent_index = []
    for i, sent in enumerate(test['text']):
        if len(sent) == 0:
            empty_sent_index.append(i)

    test = test.drop(empty_sent_index, axis = 0).reset_index(drop = True)

    print('Loaded {} train sentences'.format(len(train)))
    print('Loaded {} test sentences'.format(len(test)))
    
    return train, test
