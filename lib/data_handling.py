
import json 
import os

import pandas as pd 
import numpy as np 

import lib.globals as g

class Dataset:

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

        dataframe_path = os.path.join(g.DATA_FOLDER,os.path.splitext(from_path)[0]+'_df.pkl')

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

        df.to_pickle(dataframe_path)

        return df 



        
