import os 


DATA_FOLDER = os.path.join(os.getcwd(),"data") # directory containing the data
LOG_NAME = "LOG"  #name of the logger 

EMBEDDING_DIMENSION = 300   #vector dimension for glove embedding 

UNK_TOKEN = "[UNK]"   #unknown token string key 
PAD_TOKEN = "[PAD]"   #pad token string key 

TRAIN_VAL_SPLIT = 80         # percentage of train examples 

RND_SEED = 42 

BERT_MAX_TOKENS = 512

BERT_PRETRAINED = 'bert-base-uncased'
ELECTRA_PRETRAINED = 'google/electra-base-discriminator'