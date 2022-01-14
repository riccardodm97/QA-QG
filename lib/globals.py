import os 


DATA_FOLDER = os.path.join(os.getcwd(),"data") # directory containing the data
LOG_NAME = "LOG"  #name of the logger 

EMBEDDING_DIMENSION = 50   #vector dimension for glove embedding 

UNK_TOKEN = "[UNK]"   #unknown token string key 
PAD_TOKEN = "[PAD]"   #pad token string key 

TRAIN_VAL_SPLIT = 80         # percentage of train examples 

RND_SEED = 42 