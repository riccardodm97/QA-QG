import json
import os

import pandas as pd 
import torch

from tokenizers import  Tokenizer, Encoding
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Sequence, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.pre_tokenizers import Whitespace

from datasets import Dataset 

import lib.globals as globals
import lib.utils as utils 

class RawSquadDataset:

    JSON_RECORD = ['data','paragraphs','qas','answers']

    def __init__(self, dataset_path = None):
        
        self.dataset_path = dataset_path

        if self.dataset_path is not None:
            assert os.path.exists(self.dataset_path), 'Error : the dataset path should contain json file'

            self.df =  self._json_to_dataframe(self.dataset_path)

        else : raise Exception('The dataset path is empty') 

    
    def _json_to_dataframe(self,from_path):

        '''
        Encode the specified dataset stored as json file at 'from_path' as a Pandas Dataframe

        '''

        dataframe_path = os.path.join(globals.DATA_FOLDER,os.path.splitext(from_path)[0]+'_df.pkl')

        # If already present load dataframe from data folder 
        if os.path.exists(dataframe_path):

            df = pd.read_pickle(dataframe_path)

            self.has_labels = 'answer' in df.columns
            
            return df

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

            df['label_char'] = [x for x in zip(df['answer_start'],df['answer_end'])]

            df.drop(['answer_start','answer_end'],axis='columns',inplace=True)

            self.has_labels = True

        
        df["context_id"] = df["data.paragraph.context"].factorize()[0]

        df.rename(columns={"data.title": "title", "id":"question_id","data.paragraph.context":"context"}, inplace=True)

        df.reset_index(drop=True,inplace=True)

        # Reorder columns to be more pleaseant 
        columns=['context_id','question_id','title','context','question']
        not_pres = [col for col in df.columns if col not in columns]
        df = df[columns+not_pres]

        df = df.drop_duplicates()

        df.to_pickle(dataframe_path)

        return df 


class RecurrentDataManager:

    def __init__(self, dataset : RawSquadDataset, split : bool):

        self.raw_dataset : RawSquadDataset = dataset

        self.emb_model, self.vocab = utils.load_embedding_model()

        self.tokenizer = self._get_tokenizer()

        self.hf_dataset = self._get_hf_dataset()

    def _get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        tokenizer.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

        return tokenizer


    def _get_hf_dataset(self):  

        #encoda dataframe as Huggingface dataset 
        hf_dataset = Dataset.from_pandas(self.raw_dataset.df)

        def transform_with_label(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            starts = list(map(lambda x: x[0],batch['label_char']))
            ends = list(map(lambda x: x[1],batch['label_char']))

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings]),
                'question_ids': torch.tensor([e.ids for e in question_encodings]),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings]),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings]),
                'label_token_start': torch.tensor([e.char_to_token(starts[i]) for i,e in enumerate(context_encodings)]),
                'label_token_end': torch.tensor([e.char_to_token(ends[i]-1) for i,e in enumerate(context_encodings)])        
            }

            return batch
        
        def transform_no_label(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings]),
                'question_ids': torch.tensor([e.ids for e in question_encodings]),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings]),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings])
            }

            return batch
        
        if self.raw_dataset.has_labels :
            hf_dataset.set_transform(transform_with_label,output_all_columns=False)
        else:
            hf_dataset.set_transform(transform_no_label,output_all_columns=False)
        
        return hf_dataset
        





        
