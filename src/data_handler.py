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

import src.globals as globals
import src.utils as utils 

logger = logging.getLogger(globals.LOG_NAME)

class RawSquadDataset:

    JSON_RECORD = ['data','paragraphs','qas','answers']

    def __init__(self, train_dataset_path = None, test_dataset_path = None):

        assert train_dataset_path or test_dataset_path, 'No path has been passed'
        
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path

        self.train_df = None
        if self.train_dataset_path is not None:
            assert os.path.exists(self.train_dataset_path), 'Error : the train dataset path should contain dataset json file'

            logger.info('loading train dataset from data folder')
            self.train_df =  self._json_to_dataframe(self.train_dataset_path)
            assert "answer" in self.train_df.columns, 'Error : no answers in train dataset'

        self.test_df = None
        if self.test_dataset_path is not None:
            assert os.path.exists(self.test_dataset_path), 'Error : the test dataset path should contain dataset json file'

            logger.info('loading test dataset from data folder')
            self.test_df =  self._json_to_dataframe(self.test_dataset_path)
            self.test_has_labels = "answer" in self.test_df.columns

    
    def _json_to_dataframe(self,from_path):
        '''
        Encode the specified dataset stored as json file at 'from_path' as a Pandas Dataframe
        '''

        dataframe_path = os.path.join(globals.DATA_FOLDER,os.path.splitext(from_path)[0]+'_df.pkl')

        # If already present load dataframe from data folder 
        if os.path.exists(dataframe_path):
            logger.info('dataset as dataframe already present in data folder, loading that instead...')
            df = pd.read_pickle(dataframe_path)
            
            return df

        # Otherwise create Dataframe object 
        logger.info('creating dataframe object from json file')
        json_file = json.loads(open(from_path).read())

        df = None
        if (any(pd.json_normalize(json_file,self.JSON_RECORD[:-1]).answers.apply(len)== 0)):
            df = pd.json_normalize(json_file, self.JSON_RECORD[:-1],meta=[["data", "title"],['data','paragraph','context']])
            if "answers" in df.columns:
                df = df.drop("answers", axis="columns")
        else:
            df = pd.json_normalize(json_file , self.JSON_RECORD ,meta=[["data", "title"],['data','paragraph','context'],['data','paragraph','qas','question'],['data','paragraph','qas','id']])
            df.rename(columns={"text": "answer","data.paragraph.qas.question":"question","data.paragraph.qas.id":"id"}, inplace=True)
            df['answer_end'] = df.answer_start + df.answer.apply(len)

            df['label_char'] = [x for x in zip(df['answer_start'],df['answer_end'])]

            df.drop(['answer_start','answer_end'],axis='columns',inplace=True)

        
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

        self.dataset = dataset
        self.device = device 

        self.tokenizer = self._get_tokenizer()

        self.train_hf_dataset, self.val_hf_dataset = None, None
        if self.dataset.train_df is not None:
            train_df, val_df = self._train_val_split(self.dataset.train_df) 
            self.train_hf_dataset = self._build_hf_dataset(train_df)
            self.val_hf_dataset = self._build_hf_dataset(val_df)
        
        self.test_hf_dataset = None
        if self.dataset.test_df is  not None:
            test_df = self.dataset.test_df
            self.test_hf_dataset = self._build_hf_dataset(test_df, self.dataset.test_has_labels)
    
    def _train_val_split(self, df):

        logger.info('splitting dataset dataframe in train e val')

        df['split'] = 'train'
            
        perc_idx = int(np.percentile(df.index, globals.TRAIN_VAL_SPLIT))   #index of the row where to split 
        df.loc[df.index > perc_idx,'split'] = 'val' 

        first_val = perc_idx + 1

        c_id = df.loc[perc_idx,'context_id']

        # keep all the examples with the same context within the same split 
        for row in df[first_val:].iterrows():      

            if row[1]['context_id'] == c_id :
                df.loc[row[0],'split'] = 'train'
            else :
                break
        
        return df[df['split']=='train'], df[df['split']=='val']

    
    def get_dataloader(self, split : str, batch_size : int, random : bool):

        dataset = getattr(self,split+'_hf_dataset')
        assert dataset, f'No {split} dataset present'

        return utils.build_dataloader(dataset, batch_size, random)

        
    
    def _build_hf_dataset(self, df : pd.DataFrame, has_labels : bool = True):  

        start_time = time.perf_counter()
        logger.info('building hf_dataset')

        #encode dataframe as Huggingface dataset 
        hf_dataset = Dataset.from_pandas(df)

        hf_dataset.set_transform(self._batch_transform(has_labels),output_all_columns=False)    #TODO output_all_columns

        end_time = time.perf_counter()
        logger.info('elapsed time in building hf_dataset : %f',end_time-start_time)
        
        return hf_dataset

    
    def _get_tokenizer(self) -> Tokenizer:

        raise NotImplementedError()
    
    def _batch_transform(self, has_label) -> Callable:

        raise NotImplementedError()


    

class RecurrentDataManager(DataManager):

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        start_time = time.perf_counter()
        logger.info('init RecurrentDataManager')

        self.emb_model, self.vocab = utils.load_embedding_model()    #loading embedding model first since it's needed for the tokenizer 
        super().__init__(dataset,device)

        end_time = time.perf_counter()
        logger.info('elapsed time in building DataManager : %f',end_time-start_time)


    def _get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        tokenizer.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

        return tokenizer


    def _batch_transform(self, has_label : bool):

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
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device),
                'context_offsets': torch.tensor([e.offsets for e in context_encodings]),
                'context_text': batch['context'] 
            }

            return batch
        
        return transform_with_label if has_label else transform_no_label