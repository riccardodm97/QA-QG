import json
import os
import logging 
import time 
from typing import Callable

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
from tokenizers import  Tokenizer, Encoding
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Strip, StripAccents, NFD, BertNormalizer
from tokenizers.normalizers import Sequence as NormSequence
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
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
            self.test_has_answers = "answer" in self.test_df.columns

    
    def _json_to_dataframe(self,from_path):
        '''
        Encode the specified dataset stored as json file at 'from_path' as a Pandas Dataframe
        '''
        file_name = os.path.splitext(os.path.basename(from_path))[0]+'_df.pkl'
        dataframe_path = os.path.join(globals.DATA_FOLDER, file_name)

        # If already present load dataframe from data folder 
        if os.path.exists(dataframe_path):
            logger.info('dataset as dataframe already present in data folder, loading that instead...')
            df = pd.read_pickle(dataframe_path)
            
            return df

        # Otherwise create Dataframe object 
        logger.info('creating dataframe object from json file')
        json_file = json.loads(open(from_path).read())

        df = None
        
        answ = pd.json_normalize(json_file,self.JSON_RECORD[:-1]).answers
        if any(answ.apply(len)== 0) or answ.isnull().values.any():
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

        self._get_tokenizer() #construct the tokenizer for the current DataManager

        self.train_hf_dataset, self.val_hf_dataset = None, None
        if self.dataset.train_df is not None:
            train_df, val_df = self._train_val_split(self.dataset.train_df) 
            self.train_hf_dataset = self._build_hf_dataset(train_df)
            self.val_hf_dataset = self._build_hf_dataset(val_df)
        
        self.test_hf_dataset = None
        if self.dataset.test_df is not None:
            test_df = self.dataset.test_df
            self.test_hf_dataset = self._build_hf_dataset(test_df, self.dataset.test_has_answers)
    
    def _train_val_split(self, df):

        logger.info('splitting dataset dataframe in train e val')

        df['split'] = 'train'
            
        perc_idx = int(np.percentile(df.index, globals.TRAIN_VAL_SPLIT))   #index of the row where to split 
        df.loc[df.index > perc_idx,'split'] = 'val' 

        first_val = perc_idx + 1

        title = df.loc[perc_idx,'title']

        # keep all the examples with the same title within the same split 
        for row in df[first_val:].iterrows():      

            if row[1]['title'] == title :
                df.loc[row[0],'split'] = 'train'
            else :
                break
        
        return df[df['split']=='train'], df[df['split']=='val']

    
    def get_dataloader(self, split : str, batch_size : int, random : bool = False):

        dataset = getattr(self,split+'_hf_dataset')
        assert dataset, f'No {split} dataset present'

        if random : 
            sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)
        else:
            sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    
        return DataLoader(dataset,sampler=sampler,batch_size=None,num_workers=4)  #TODO num_workers 
        
    
    def _build_hf_dataset(self, df : pd.DataFrame, has_answers : bool = True):  

        start_time = time.perf_counter()
        logger.info('building hf_dataset')

        #encode dataframe as Huggingface dataset 
        hf_dataset = Dataset.from_pandas(df)

        hf_dataset.set_transform(self._batch_transform(has_answers))

        end_time = time.perf_counter()
        logger.info('elapsed time in building hf_dataset : %f',end_time-start_time)
        
        return hf_dataset

    
    def _get_tokenizer(self) :

        raise NotImplementedError()
    
    def _batch_transform(self, has_answer) -> Callable:

        raise NotImplementedError()



class RecurrentDataManager(DataManager):

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        start_time = time.perf_counter()
        logger.info('init RecurrentDataManager')

        self.emb_model, self.vocab = utils.get_Glove_model_and_vocab()    #loading embedding model first since it's needed for the tokenizer 
        super().__init__(dataset,device)

        end_time = time.perf_counter()
        logger.info('elapsed time in building DataManager : %f',end_time-start_time)


    def _get_tokenizer(self):

        tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        tokenizer.normalizer = NormSequence([NFD(), StripAccents(), Lowercase(), Strip()]) #BertNormalizer() 
        tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        tokenizer.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

        self.tokenizer = tokenizer


    def _batch_transform(self, has_answer : bool):

        def transform_with_answer(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            starts = list(map(lambda x: x[0],batch['label_char']))
            ends = list(map(lambda x: x[1],batch['label_char']))

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings], device=self.device),
                'question_ids': torch.tensor([e.ids for e in question_encodings], device=self.device),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings], device=self.device),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device),
                'offsets': torch.tensor([e.offsets for e in context_encodings]), 
                'label_token_start': torch.tensor([e.char_to_token(starts[i]) for i,e in enumerate(context_encodings)], device=self.device),
                'label_token_end': torch.tensor([e.char_to_token(ends[i]-1) for i,e in enumerate(context_encodings)], device=self.device),
                'context_text': batch['context'],
                'answer_text': batch['answer'],
                'question_alpha' : batch['question_id']  
            }

            return batch
        
        def transform_no_answer(batch):

            context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
            question_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['question'])

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings], device=self.device),
                'question_ids': torch.tensor([e.ids for e in question_encodings], device=self.device),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings], device=self.device),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device),
                'offsets': torch.tensor([e.offsets for e in context_encodings]),
                'context_text': batch['context'],
                'question_alpha' : batch['question_id']
            }

            return batch
        
        return transform_with_answer if has_answer else transform_no_answer


class TransformerDataManager(DataManager):

    VOCAB_PATH = os.path.join(globals.DATA_FOLDER,globals.BERT_PRETRAINED+'-vocab.txt')

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        start_time = time.perf_counter()
        logger.info('init TransformerDataManager')

        super().__init__(dataset,device)

        end_time = time.perf_counter()
        logger.info('elapsed time in building DataManager : %f',end_time-start_time)


    def _get_tokenizer(self):

        if not os.path.exists(self.VOCAB_PATH):
            utils.load_bert_vocab()

        tokenizer = BertWordPieceTokenizer(self.VOCAB_PATH, lowercase=True)
        tokenizer.enable_padding(direction="right", pad_type_id=1)
        tokenizer.enable_truncation(globals.BERT_MAX_TOKENS, strategy='only_second', stride = 25)

        self.tokenizer = tokenizer


    def _batch_transform(self, has_answer : bool):

        def transform_with_answer(batch):

            encodings: list[Encoding] = self.tokenizer.encode_batch(list(zip(batch['question'],batch['context'])))

            starts = list(map(lambda x: x[0],batch['label_char']))
            ends = list(map(lambda x: x[1],batch['label_char']))

            not_replaced = []
            for i,e in enumerate(encodings) :
                if e.char_to_token(starts[i],1) is None or e.char_to_token(ends[i]-1,1) is None :
                    flag = 0
                    for o in e.overflowing :
                        if o.char_to_token(starts[i],1) is not None and o.char_to_token(ends[i]-1,1) is not None :
                            encodings[i] = o
                            flag = 1
                            break
                    if flag==0 :
                        not_replaced.append(i)
            
            for idx in sorted(not_replaced, reverse = True):
                encodings.pop(idx)
                starts.pop(idx)
                ends.pop(idx)
                batch['context'].pop(idx)
                batch['answer'].pop(idx)

            batch = {
                'ids': torch.tensor([e.ids for e in encodings], device=self.device),
                'mask': torch.tensor([e.attention_mask for e in encodings], device=self.device),
                'special_tokens_mask':torch.tensor([e.special_tokens_mask for e in encodings], device=self.device),
                'offsets': torch.tensor([e.offsets for e in encodings], device=self.device), 
                'type_ids': torch.tensor([e.type_ids for e in encodings], device=self.device),
                'label_token_start': torch.tensor([e.char_to_token(starts[i],1) for i,e in enumerate(encodings)], device=self.device),
                'label_token_end': torch.tensor([e.char_to_token(ends[i]-1,1) for i,e in enumerate(encodings)], device=self.device),
                'context_text': batch['context'],
                'answer_text': batch['answer'],
                'question_alpha' : batch['question_id']
            }

            return batch

        
        def transform_no_answer(batch):

            encodings: list[Encoding] = self.tokenizer.encode_batch(list(zip(batch['question'],batch['context'])))

            batch = {
                'ids': torch.tensor([e.ids for e in encodings], device=self.device),
                'mask': torch.tensor([e.attention_mask for e in encodings], device=self.device),
                'special_tokens_mask':torch.tensor([e.special_tokens_mask for e in encodings], device=self.device),
                'offsets': torch.tensor([e.offsets for e in encodings], device=self.device), 
                'type_ids': torch.tensor([e.type_ids for e in encodings], device=self.device),
                'context_text': batch['context'],
                'question_alpha' : batch['question_id']
            }

            return batch
        
        return transform_with_answer if has_answer else transform_no_answer


class QGDataManager(DataManager):

    def __init__(self, dataset : RawSquadDataset, device = 'cpu'):

        start_time = time.perf_counter()
        logger.info('init QGDataManager')

        super().__init__(dataset, device)

        self.enc_vectors = utils.build_embedding_matrix('encoder',self.enc_tokenizer.get_vocab())   #loading embedding model first since it's needed for the tokenizer 
        self.dec_vectors = utils.build_embedding_matrix('decoder',self.dec_tokenizer.get_vocab()) 

        end_time = time.perf_counter()
        logger.info('elapsed time in building DataManager : %f',end_time-start_time)


    def _get_tokenizer(self):

        #post processor template 
        processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] [SOS]:1 $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", 2),
                ("[EOS]", 3),
            ],
        )

        #build the two tokenizers (encoder and decoder)
        enc_tokenizer = Tokenizer(WordLevel(unk_token=globals.UNK_TOKEN))
        enc_tokenizer.normalizer = BertNormalizer(handle_chinese_chars=False) 
        enc_tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        enc_tokenizer.post_processor = processor 

        dec_tokenizer = Tokenizer(WordLevel(unk_token=globals.UNK_TOKEN))
        dec_tokenizer.normalizer = BertNormalizer(handle_chinese_chars=False) 
        dec_tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        dec_tokenizer.post_processor = processor 

        #trainers for fitting the two tokenizers on the dataset
        enc_trainer = WordLevelTrainer(special_tokens=[globals.PAD_TOKEN,globals.UNK_TOKEN,globals.SOS_TOKEN,globals.EOS_TOKEN],vocab_size=65000)  #TODO size 
        dec_trainer = WordLevelTrainer(special_tokens=[globals.PAD_TOKEN,globals.UNK_TOKEN,globals.SOS_TOKEN,globals.EOS_TOKEN],vocab_size=40000)   

        #which part of the dataset each tokenizer focuses on 
        enc_text = self.dataset.train_df.context.to_list() + self.dataset.train_df.answer.to_list()
        dec_text = self.dataset.train_df.question.to_list()

        enc_tokenizer.train_from_iterator(enc_text,trainer=enc_trainer) 
        enc_tokenizer.enable_padding(direction="right", pad_id=enc_tokenizer.token_to_id(globals.PAD_TOKEN), pad_type_id=1, pad_token=globals.PAD_TOKEN)

        dec_tokenizer.train_from_iterator(dec_text,trainer=dec_trainer) 
        dec_tokenizer.enable_padding(direction="right", pad_id=enc_tokenizer.token_to_id(globals.PAD_TOKEN), pad_type_id=1, pad_token=globals.PAD_TOKEN)

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer      #TODO clean text before ? 


    def _batch_transform(self, has_answer : bool):

        assert has_answer, 'Answers are needed for Question Generation task'

        def transform(batch):

            context_encodings: list[Encoding] = self.enc_tokenizer.encode_batch(batch['context'])
            answer_encodings : list[Encoding] = self.enc_tokenizer.encode_batch(batch['answer'])
            question_encodings: list[Encoding] = self.dec_tokenizer.encode_batch(batch['question'])

            starts = list(map(lambda x: x[0],batch['label_char']))
            ends = list(map(lambda x: x[1],batch['label_char']))

            batch = {
                'context_ids': torch.tensor([e.ids for e in context_encodings], device=self.device),
                'answer_ids': torch.tensor([e.ids for e in answer_encodings], device=self.device),
                'question_ids': torch.tensor([e.ids for e in question_encodings], device=self.device),
                'context_mask': torch.tensor([e.attention_mask for e in context_encodings], device=self.device),
                'answer_mask': torch.tensor([e.attention_mask for e in answer_encodings], device=self.device),
                'question_mask': torch.tensor([e.attention_mask for e in question_encodings], device=self.device),
                'answer_token_start': torch.tensor([e.char_to_token(starts[i],1) for i,e in enumerate(context_encodings)], device=self.device),
                'answer_token_end': torch.tensor([e.char_to_token(ends[i]-1,1) for i,e in enumerate(context_encodings)], device=self.device),
            }

            return batch
    
        
        return transform
