
import json
import os
import logging 
import time 

from typing import Callable

import numpy as np
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

logger = logging.getLogger(globals.LOG_NAME)

class RawSquadDataset:

    JSON_RECORD = ['data','paragraphs','qas','answers']

    def __init__(self, dataset_path = None):
        
        self.dataset_path = dataset_path

        if self.dataset_path is not None:
            assert os.path.exists(self.dataset_path), 'Error : the dataset path should contain dataset json file'

            logger.info('loading dataset from data folder')
            self.df =  self._json_to_dataframe(self.dataset_path)

        else : raise Exception('The dataset path is empty') 

    
    def _json_to_dataframe(self,from_path):

        '''
        Encode the specified dataset stored as json file at 'from_path' as a Pandas Dataframe

        '''

        dataframe_path = os.path.join(globals.DATA_FOLDER,os.path.splitext(from_path)[0]+'_df.pkl')

        # If already present load dataframe from data folder 
        if os.path.exists(dataframe_path):
            logger.info('dataset as dataframe already present in data folder, loading that instead...')
            df = pd.read_pickle(dataframe_path)
            self.has_labels = 'answer' in df.columns
            
            return df

        # Otherwise create Dataframe object 
        logger.info('creating dataframe object from json file')
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

        logger.info('saving dataframe in data folder as pickle file')
        df.to_pickle(dataframe_path)
        logger.info('saved')

        return df 

class DataManager: 

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        self.df = dataset.df.copy()
        self.device = device 

        self.tokenizer = self._get_tokenizer()

        train_df, val_df = self._train_val_split() 
        self.train_hf_dataset = self._build_hf_dataset(train_df)
        self.val_hf_dataset = self._build_hf_dataset(val_df)
    
    def _train_val_split(self):

        logger.info('splitting dataset dataframe in train e val')

        self.df['split'] = 'train'
            
        perc_idx = int(np.percentile(self.df.index, globals.TRAIN_VAL_SPLIT))   #index of the row where to split 
        self.df.loc[self.df.index > perc_idx,'split'] = 'val' 

        first_val = perc_idx + 1

        c_id = self.df.loc[perc_idx,'context_id']

        # keep all the examples with the same context within the same split 
        for row in self.df[first_val:].iterrows():      

            if row[1]['context_id'] == c_id :
                self.df.loc[row[0],'split'] = 'train'
            else :
                break
        
        return self.df[self.df['split']=='train'], self.df[self.df['split']=='val']

    
    def get_dataloaders(self,batch_size : int, random : bool):

        #TODO assicurarsi che ci sono gli hf_dataset 
        
        train_dl = utils.build_dataloader(self.train_hf_dataset, batch_size, random)
        val_dl = utils.build_dataloader(self.val_hf_dataset, batch_size, random)

        return train_dl, val_dl 

    
    def _build_hf_dataset(self,df):  

        start = time.perf_counter()
        logger.info('building one hf_dataset')

        #encode dataframe as Huggingface dataset 
        hf_dataset = Dataset.from_pandas(df)

        hf_dataset.set_transform(self._batch_transform(),output_all_columns=False)    #TODO output_all_columns

        end = time.perf_counter()
        logger.info('elapsed time in building hf_dataset : %f',start-end)
        
        return hf_dataset

    
    def _get_tokenizer(self) -> Tokenizer:

        raise NotImplementedError()
    
    def _batch_transform(self, label: bool) -> Callable:

        raise NotImplementedError()


    

class RecurrentDataManager(DataManager):

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        start = time.perf_counter()
        logger.info('init RecurrentDataManager')

        self.emb_model, self.vocab = utils.load_embedding_model()    #loading embedding model first since it's needed for the tokenizer 
        super().__init__(dataset,device)

        end = time.perf_counter()
        logger.info('elapsed time in building DataManager : %f',start-end)


    def _get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        tokenizer.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

        return tokenizer


    def _batch_transform(self, label : bool):

        def transform_with_label(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            starts = list(map(lambda x: x[0],batch['label_char']))
            ends = list(map(lambda x: x[1],batch['label_char']))

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings], device=self.device),
                'question_ids': torch.tensor([e.ids for e in question_encodings], device=self.device),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings], device=self.device),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device),
                'context_offsets': torch.tensor([e.offsets for e in context_encodings]), 
                'label_token_start': torch.tensor([e.char_to_token(starts[i]) for i,e in enumerate(context_encodings)], device=self.device),
                'label_token_end': torch.tensor([e.char_to_token(ends[i]-1) for i,e in enumerate(context_encodings)], device=self.device),
                'context_text': batch['context'],
                'answer_text': batch['answer']    
            }

            return batch
        
        def transform_no_label(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings], device=self.device),
                'question_ids': torch.tensor([e.ids for e in question_encodings], device=self.device),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings], device=self.device),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device)
            }

            return batch
        
        return transform_with_label if label else transform_no_label