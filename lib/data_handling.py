
from glob import glob
import json
import os

import pandas as pd 

from tokenizers import  Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Sequence, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.pre_tokenizers import Whitespace

from datasets import Dataset 

import lib.globals as glob
import lib.utils as utils 

class RawSquadDataset:

    JSON_RECORD = ['data','paragraphs','qas','answers']

    def __init__(self, dataset_path = None):
        
        self.dataset_path = dataset_path

        if self.dataset_path is not None:
            assert os.path.exists(self.dataset_path), 'Error : the dataset path should contain json file'

            self.raw_df =  self._json_to_dataframe(self.dataset_path)

        else : raise Exception('The dataset path is empty') 

    
    def _json_to_dataframe(self,from_path):

        '''
        Encode the specified dataset stored as json file at 'from_path' as a Pandas Dataframe

        '''

        dataframe_path = os.path.join(glob.DATA_FOLDER,os.path.splitext(from_path)[0]+'_df.pkl')

        # If already present load dataframe from data folder 
        if os.path.exists(dataframe_path):
            return pd.read_pickle(dataframe_path)

        # Otherwise create Dataframe object 

        json_file = json.loads(open(from_path).read())

        df = None
        if (any(pd.json_normalize(json_file,self.JSON_RECORD[:-1]).answers.apply(len)== 0)):
            df = pd.json_normalize(json_file, self.JSON_RECORD[:-1],meta=[["data", "title"],['data','paragraph','context']])
            if "answers" in df.columns:
                df = df.drop("answers", axis="columns")
            self.has_labels = False
        else:
            df = pd.json_normalize(json_file , self.JSON_RECORD ,meta=[["data", "title"],['data','paragraph','context'],['data','paragraph','qas','question'],['data','paragraph','qas','id']])
            df.rename(columns={"text": "answer","data.paragraph.qas.question":"question","data.paragraph.qas.id":"id"}, inplace=True)
            df['answer_end'] = df.answer_start + df.answer.apply(len)
            self.has_labels = True

        
        df["context_id"] = df["data.paragraph.context"].factorize()[0]

        df.rename(columns={"data.title": "title", "id":"question_id","data.paragraph.context":"context"}, inplace=True)

        df.reset_index(drop=True,inplace=True)

        # Reorder columns to be more pleaseant 
        columns=['title','context','context_id','question','question_id']
        not_pres = [col for col in df.columns if col not in columns]
        df = df[columns+not_pres]

        df = df.drop_duplicates()

        df.to_pickle(dataframe_path)

        return df 


class DataManager:

    def __init__(self, dataset : RawSquadDataset):

        self.raw_dataset = dataset

        self.emb_model, self_vocab = utils.load_embedding_model()

        self.tokenizer = self._get_tokenizer()

        self.hf_dataset = self._tokenize_and_qualcosa()

    def _get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=glob.UNK_TOKEN))
        tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])

        return tokenizer


    def _tokenize_and_qualcosa(self):  #TODO cambiare nome

        #encoda dataframe as Huggingface dataset 
        hf_dataset = Dataset.from_pandas(self.raw_dataset.raw_df)

        def _convert_to_feature(batch):
        
            context_encodings = self.tokenizer.encode_batch(batch['context'])
            question_encodings = self.tokenizer.encode(batch['question'])

            
            encodings = {
                'input_ids': [e.ids for e in context_encodings], 
                'attention_mask': [e.attention_mask for e in context_encodings],
            }

            return encodings    
        
        def _add_start_end_token(batch):

            return 

        def _on_the_fly_transform(batch):
            self.tokenizer.enable_padding(direction="right", pad_id=self.vocab['[PAD]'], pad_type_id=0, pad_token=glob.PAD_TOKEN)
            padded_encodings = self.tokenizer.encode_batch(batch['context'])

            encodings = {
                'padded_ids': [e.ids for e in padded_encodings], 
                'context': batch['context']
            }

            return encodings

        
        #tokenize the entire dataset and add specific columns to hf_dataset
        hf_dataset = hf_dataset.map(_convert_to_feature,batched=True)

        #insert intto hf_dataset start and end index wrt to context in the token space 
        hf_dataset = hf_dataset.map(_add_start_end_token)

        if self.raw_dataset.has_labels :

            #fai roba TODO

            return 


        
        return hf_dataset
        





        
