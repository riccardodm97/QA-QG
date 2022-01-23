import torch
from torch import nn

from transformers import BertModel, ElectraModel

import src.layer as layer 
import src.globals as globals 
from src.utils import get_embedding_layer

class DrQA(nn.Module):
    
    def __init__(self, hidden_dim, num_layers, dropout, weights_matrix, pad_idx, device):
        
        super().__init__()
        
        self.device = device      
        self.num_directions = 2   #bidirectional LSTMs
        
        self.embedding_layer, self.embedding_dim = get_embedding_layer(weights_matrix, pad_idx, device)

        # self.question_bilstm = layer.StackedBiLSTM(self.embedding_dim, hidden_dim, num_layers, dropout)
        # self.linear_attn_question = layer.LinearAttentionLayer(hidden_dim*num_layers*self.num_directions) 

        self.context_bilstm = layer.StackedBiLSTM(self.embedding_dim * 2, hidden_dim, num_layers, dropout)
        
        self.align_embedding = layer.AlignQuestionEmbedding(self.embedding_dim)
        
        self.question_encoding = layer.QuestionEncodingLayer(self.embedding_dim,hidden_dim,num_layers,dropout)
        
        self.bilinear_attn_start = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions, hidden_dim*num_layers*self.num_directions)
        
        self.bilinear_attn_end = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions,hidden_dim*num_layers*self.num_directions)
        
        self.dropout = nn.Dropout(dropout)

        self.to(self.device)
    
    def get_model_name(self) -> str :
        return 'DrQA'
      
    
    def forward(self, inputs):   
       
        context = inputs['context_ids']                               # [bs, len_c]
        question = inputs['question_ids']                             # [bs, len_q]
        context_mask = inputs['context_mask']                         # [bs, len_c]
        question_mask = inputs['question_mask']                       # [bs, len_q]
        context_lengths = torch.count_nonzero(context_mask,dim=1)     # [bs]
        question_lengths = torch.count_nonzero(question_mask,dim=1)   # [bs]
        
        ctx_embed = self.embedding_layer(context)
        # ctx_embed = [bs, len_c, emb_dim]
        
        qst_embed = self.embedding_layer(question)
        # ques_embed = [bs, len_q, emb_dim]

        ctx_embed = self.dropout(ctx_embed)
     
        qst_embed = self.dropout(qst_embed)
             
        align_embed = self.align_embedding(ctx_embed, qst_embed, question_mask)
        # align_embed = [bs, len_c, emb_dim]  
        
        ctx_bilstm_input = torch.cat([ctx_embed, align_embed], dim=2)
        # ctx_bilstm_input = [bs, len_c, emb_dim*2]
                
        ctx_encoded = self.context_bilstm(ctx_bilstm_input, context_lengths)
        # ctx_outputs = [bs, len_c, hid_dim*layers*dir] = [bs, len_c, hid_dim*6]
       
        # qst_outputs = self.question_bilstm(qst_embed, question_lengths)
        # # qtn_outputs = [bs, len_q, hid_dim*6]
    
        # qst_weights = self.linear_attn_question(qst_outputs, question_mask)
        # # qtn_weights = [bs, len_q]
            
        # qst_encoded = layer.LinearAttentionLayer.weighted_average(qst_outputs, qst_weights)
        # # qtn_weighted = [bs, hid_dim*6]

        qst_encoded = self.question_encoding(qst_embed, question_mask, question_lengths)
        # qtn_weighted = [bs, hid_dim*6]

        start_scores = self.bilinear_attn_start(ctx_encoded, qst_encoded, context_mask)
        # start_scores = [bs, len_c]
         
        end_scores = self.bilinear_attn_end(ctx_encoded, qst_encoded, context_mask)
        # end_scores = [bs, len_c]
        
      
        return start_scores, end_scores
    
class BertQA(nn.Module):

    def __init__(self, device, dropout = 0.1) :      #TODO qualcos'altro ?
        super().__init__()

        self.device = device

        self.bert = BertModel.from_pretrained(globals.BERT_PRETRAINED)
        self.start_token_classifier =  nn.Linear(self.bert.config.hidden_size, 1)
        self.end_token_classifier =  nn.Linear(self.bert.config.hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

        self.to(device)
    
    def get_model_name(self) -> str :
        return 'BertQA'
    

    def forward(self, inputs):   
       
        ids = inputs['ids']                           # [bs, len_text]
        mask = inputs['mask']                         # [bs, len_text]
        type_ids = inputs['type_ids']                 # [bs, len_text]
        special_token_mask = inputs['special_tokens_mask']        
        answer_space_mask = type_ids & ~special_token_mask           

        bert_outputs = self.bert(input_ids = ids, attention_mask = mask, token_type_ids = type_ids)

        sequence_outputs = self.dropout(bert_outputs[0])
        # [bs, len_txt, bert_hidden_dim]

        start_scores = self.start_token_classifier(sequence_outputs)  
        end_scores = self.end_token_classifier(sequence_outputs)
        # [bs, len_txt, 1]

        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)
        # [bs, len_txt]

        start_scores = start_scores.masked_fill(answer_space_mask == 0, float('-inf'))
        end_scores = end_scores.masked_fill(answer_space_mask == 0, float('-inf'))

        return start_scores, end_scores


class ElectraQA(nn.Module):

    def __init__(self, device, dropout, hidden_dim, freeze) :      #TODO qualcos'altro ?
        super().__init__()

        self.device = device

        self.electra = ElectraModel.from_pretrained(globals.ELECTRA_PRETRAINED)
        self.rnn = nn.LSTM(self.electra.config.hidden_size, hidden_dim, batch_first = True, bidirectional = True)
        self.token_classifier =  nn.Linear(hidden_dim*2, 2)

        self.dropout = nn.Dropout(dropout)

        if freeze : 
            for param in self.electra.parameters():
                param.requires_grad = False

        self.to(device)
    
    def get_model_name(self) -> str :
        return 'ElectraQA'
    
    def forward(self, inputs):   
       
        ids = inputs['ids']                           # [bs, len_text]
        mask = inputs['mask']                         # [bs, len_text]
        type_ids = inputs['type_ids']                 # [bs, len_text]
        special_token_mask = inputs['special_tokens_mask']        
        answer_space_mask = type_ids & ~special_token_mask           

        electra_outputs = self.electra(input_ids = ids, attention_mask = mask, token_type_ids = type_ids)

        sequence_outputs = self.dropout(electra_outputs[0])
        # [bs, len_txt, electra_hidden_dim]

        lstm_out, _  = self.rnn(sequence_outputs)
        # [bs, len_txt, lstm_hidden_dim*2]

        token_scores = self.token_classifier(lstm_out)  
        # [bs, len_txt, 2]

        start_scores, end_scores = token_scores.split(1, dim=-1)

        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)
        # [bs, len_txt]

        start_scores = start_scores.masked_fill(answer_space_mask == 0, float('-inf'))
        end_scores = end_scores.masked_fill(answer_space_mask == 0, float('-inf'))

        return start_scores, end_scores



