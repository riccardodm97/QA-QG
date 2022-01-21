import sys
import os 

sys.path.insert(1, os.getcwd())

import os 
import pandas as pd
import torch
import numpy as np

import src.utils as utils 
import src.globals as globals
import src.data_handler as handler

from tokenizers import  Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Sequence, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
from tokenizers import Encoding
from datasets import Dataset 
from tqdm import tqdm

class BuildErrorsDf:
    def __init__(self, dataset_path):

        self.error_ids = open(os.path.join(globals.DATA_FOLDER,'error_ids.txt')).read().splitlines()
        self.dataframe_path_to_save = os.path.join(globals.DATA_FOLDER, 'error_df.csv')
        squad_dataset = handler.RawSquadDataset(dataset_path)
        self.df = squad_dataset.train_df.copy()
        self.hf_dataset_rnn = Dataset.from_pandas(squad_dataset.train_df)
        self.hf_dataset_transformer = Dataset.from_pandas(squad_dataset.train_df)
        self.model, self.vocab = utils.load_embedding_model()

        self.tokenizer_rnn = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        self.tokenizer_rnn.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        self.tokenizer_rnn.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        self.tokenizer_rnn.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

        VOCAB_PATH = os.path.join(globals.DATA_FOLDER,globals.BERT_PRETRAINED+'-vocab.txt')
        if not os.path.exists(VOCAB_PATH):
            utils.load_bert_vocab()

        self.tokenizer_transformer = BertWordPieceTokenizer(VOCAB_PATH, lowercase=True)
        self.tokenizer_transformer.enable_padding(direction="right", pad_type_id=1)
        self.tokenizer_transformer.enable_truncation(globals.BERT_MAX_TOKENS, strategy='only_second', stride = 25)

    def transform_rnn(self, batch):

        context_encodings: list[Encoding] = self.tokenizer_rnn.encode_batch(batch['context'])
        answer_encodings: list[Encoding] = self.tokenizer_rnn.encode_batch(batch['answer'])
        
        starts = list(map(lambda x: x[0],batch['label_char']))
        ends = list(map(lambda x: x[1],batch['label_char']))

        encodings = {
            'offsets': [e.offsets for e in context_encodings], 
            'context_text': batch['context'],
            'question_text': batch['question'],
            'answer_text': batch['answer'],
            'token_start_rnn': [e.char_to_token(starts[i]) for i,e in enumerate(context_encodings)],
            'token_end_rnn': [e.char_to_token(ends[i]-1) for i,e in enumerate(context_encodings)],
            'char_start_rnn': starts,
            'char_end_rnn': ends,
            'question_id' : batch['question_id'],
            'context_tokens': [e.tokens for e in context_encodings], 
            'answer_tokens': [e.tokens for e in answer_encodings],
        }

        return encodings

    def transform_transformer(self, batch):

            encodings: list[Encoding] = self.tokenizer_transformer.encode_batch(list(zip(batch['question'],batch['context'])))

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
            
            for idx in not_replaced:
                encodings.pop(idx)
                starts.pop(idx)
                ends.pop(idx)
                batch['context'].pop(idx)
                batch['answer'].pop(idx)

            batch = {
                'offsets': [e.offsets for e in encodings], 
                'type_ids': [e.type_ids for e in encodings],
                'token_start_trans': [e.char_to_token(starts[i],1) for i,e in enumerate(encodings)],
                'token_end_trans': [e.char_to_token(ends[i]-1,1) for i,e in enumerate(encodings)],
                'context_text': batch['context'],
                'question_text': batch['question'],
                'answer_text': batch['answer'],
                'char_start_trans' : starts,
                'char_end_trans': ends,
                'question_id' : batch['question_id']
            }

            return batch
    
    def build_error_dataframe_rnn(self):
        error_df_total = pd.DataFrame(columns=['context_text', 'question_text', 'answer_text', 'token_start_rnn', 
                                        'token_end_rnn', 'char_start_rnn', 'char_end_rnn','question_id'])

        for i in tqdm(range(self.df.shape[0])):

            ex = self.hf_dataset_rnn[i]
            start_c = ex['char_start_rnn']
            end_c = ex['char_end_rnn']
            starts, ends = zip(*ex['offsets'])
            if start_c in starts and end_c in ends:
                continue
            else:
                error_df_total = error_df_total.append(ex, ignore_index=True)

        error_df_total.drop(['offsets','context_tokens','answer_tokens'], axis=1, inplace=True)
        error_df_total['Rnn New'] = '✔'

        error_df = pd.DataFrame(columns=['context_text', 'question_text', 'answer_text', 'token_start_rnn', 
                                        'token_end_rnn', 'char_start_rnn', 'char_end_rnn','question_id'])

        for idx in self.error_ids:
            ex = self.hf_dataset_rnn[int(self.df[self.df['question_id'] == idx].index.values.astype(int)[0])]

            error_df = error_df.append(ex, ignore_index=True)

        error_df.drop(['offsets','context_tokens','answer_tokens'], axis=1, inplace=True)
        error_df['Rnn Old'] = '✔'

        return error_df_total, error_df

    def build_error_dataframe_transformer(self):
        error_df_total = pd.DataFrame(columns=['context_text', 'question_text', 'answer_text', 'token_start_trans', 
                                        'token_end_trans', 'char_start_trans', 'char_end_trans','question_id'])
                                        
        for i in tqdm(range(self.df.shape[0])):

            ex = self.hf_dataset_transformer[i]

            offsets = np.array(ex['offsets'])[np.array(ex['type_ids'], dtype=bool)]

            start_c = ex['char_start_trans']
            end_c = ex['char_end_trans']

            starts, ends = zip(*offsets)

            if start_c in starts and end_c in ends:
                continue
            else:
                error_df_total = error_df_total.append(ex, ignore_index=True)
        
        error_df_total.drop(['offsets','type_ids'], axis=1, inplace=True)
        error_df_total['Transformer'] = '✔'

        return error_df_total

    def find_errors(self):
        self.hf_dataset_rnn.set_transform(self.transform_rnn,output_all_columns=False)
        self.hf_dataset_transformer.set_transform(self.transform_transformer,output_all_columns=False)
        error_df_total_rnn, error_df_rnn = self.build_error_dataframe_rnn()
        error_df_total_transformer = self.build_error_dataframe_transformer()
        error_df_total_rnn = error_df_total_rnn.merge(error_df_rnn, how='outer')
        error_df_total = error_df_total_rnn.merge(error_df_total_transformer, how='outer', on=['question_id','context_text','answer_text','question_text'])
        error_df_total.to_csv(self.dataframe_path_to_save)

if __name__ == '__main__':
    dataset_path = os.path.join(globals.DATA_FOLDER,'training_set.json')
    preprocessing = BuildErrorsDf(dataset_path)
    preprocessing.find_errors()