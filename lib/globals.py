import os 


DATA_FOLDER = os.path.join(os.getcwd(),"data") # directory containing the data

EMBEDDING_DIMENSION = 50   #vector dimension for embedding matrix  

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"

TRAIN_VAL_SPLIT = 80         # percentage of train examples 