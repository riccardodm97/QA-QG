import sys
import os 

sys.path.insert(1, os.getcwd())

import os 
import pandas as pd

import src.utils as utils 
import src.globals as globals
import src.data_handler as handler


from tokenizers import  Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Sequence, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Encoding
from datasets import Dataset 
from tqdm import tqdm

class BuildErrorsDf:
    def __init__(self, dataset_path):

        self.error_ids = open(os.path.join(globals.DATA_FOLDER,'error_ids.txt')).read().splitlines()
        self.dataframe_path_to_save = os.path.join(globals.DATA_FOLDER, 'error_df.csv')
        squad_dataset = handler.RawSquadDataset(dataset_path)
        self.df = squad_dataset.train_df.copy()
        self.hf_dataset = Dataset.from_pandas(squad_dataset.train_df)
        self.model, self.vocab = utils.load_embedding_model()

        self.tokenizer = Tokenizer(WordLevel(self.vocab,unk_token=globals.UNK_TOKEN))
        self.tokenizer.normalizer = Sequence([StripAccents(), Lowercase(), Strip()])
        self.tokenizer.pre_tokenizer = PreSequence([Whitespace(), Punctuation()])
        self.tokenizer.enable_padding(direction="right", pad_id=self.vocab[globals.PAD_TOKEN], pad_type_id=1, pad_token=globals.PAD_TOKEN)

    def transform(self, batch):

        context_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['context'])
        answer_encodings: list[Encoding] = self.tokenizer.encode_batch(batch['answer'])
        
        starts = list(map(lambda x: x[0],batch['label_char']))
        ends = list(map(lambda x: x[1],batch['label_char']))

        encodings = {
            'offsets': [e.offsets for e in context_encodings], 
            'context_text': batch['context'],
            'question_text': batch['question'],
            'answer_text': batch['answer'],
            'label_token_start': [e.char_to_token(starts[i]) for i,e in enumerate(context_encodings)],
            'label_token_end': [e.char_to_token(ends[i]-1) for i,e in enumerate(context_encodings)],
            'label_char_start': starts,
            'label_char_end': ends,
            'question_id' : batch['question_id'],
            'context_tokens': [e.tokens for e in context_encodings], 
            'answer_tokens': [e.tokens for e in answer_encodings]
        }

        return encodings
    
    def build_error_dataframe(self):
        error_df_total = pd.DataFrame(columns=['context_text', 'question_text', 'answer_text', 'label_token_start', 
                                        'label_token_end', 'label_char_start', 'label_char_end','question_id'])

        for i in tqdm(range(self.df.shape[0])):

            ex = self.hf_dataset[i]
            start_c = ex['label_char_start']
            end_c = ex['label_char_end']
            starts, ends = zip(*ex['offsets'])

            if start_c not in starts and end_c in ends:
                error_df_total = error_df_total.append(ex, ignore_index=True)
            elif start_c in starts and end_c not in ends:
                error_df_total = error_df_total.append(ex, ignore_index=True)
            elif start_c not in starts and end_c not in ends:
                error_df_total = error_df_total.append(ex, ignore_index=True)
            elif ex['context_tokens'][starts.index(start_c)] != ex['answer_tokens'][0] or ex['context_tokens'][ends.index(end_c)] != ex['answer_tokens'][-1]:
                error_df_total = error_df_total.append(ex, ignore_index=True)

        error_df_total.drop(['offsets','context_tokens','answer_tokens'], axis=1, inplace=True)
        error_df_total['New'] = '✔'

        error_df = pd.DataFrame(columns=['context_text', 'question_text', 'answer_text', 'label_token_start', 
                                        'label_token_end', 'label_char_start', 'label_char_end','question_id'])

        for idx in self.error_ids:
            ex = self.hf_dataset[int(self.df[self.df['question_id'] == idx].index.values.astype(int)[0])]

            error_df = error_df.append(ex, ignore_index=True)

        error_df.drop(['offsets','context_tokens','answer_tokens'], axis=1, inplace=True)
        error_df['Old'] = '✔'

        return error_df_total, error_df

    def find_errors(self):
        self.hf_dataset.set_transform(self.transform,output_all_columns=False)
        error_df_total, error_df = self.build_error_dataframe()
        error_df_total = error_df_total.merge(error_df, how='outer')
        error_df_total.to_csv(self.dataframe_path_to_save)

if __name__ == '__main__':
    dataset_path = os.path.join(globals.DATA_FOLDER,'training_set.json')
    preprocessing = BuildErrorsDf(dataset_path)
    preprocessing.find_errors()