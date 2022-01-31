import random
import torch
from torch import nn

from transformers import BertModel, ElectraModel

import src.layer as layer 
import src.globals as globals 

class DrQA(nn.Module):
    
    def __init__(self, hidden_dim, num_layers, dropout, freeze_emb, weights_matrix, pad_idx, device):
        
        super().__init__()
        
        self.device = device      
        self.num_directions = 2   #bidirectional LSTMs

        def tune_embedding(grad, words=1000):   #train only the first 1000 most frequent words 
            grad[words:] = 0
            return grad
        
        self.emb_layer = layer.EmbeddingLayer(weights_matrix, pad_idx, tune_embedding, dropout, freeze_emb, device)

        self.context_bilstm = layer.StackedBiLSTM(self.emb_layer.embedding_dim* 2, hidden_dim, num_layers, dropout)
        
        self.align_embedding = layer.AlignQuestionEmbedding(self.emb_layer.embedding_dim)
        
        self.question_encoding = layer.QuestionEncodingLayer(self.emb_layer.embedding_dim,hidden_dim,num_layers,dropout)
        
        self.bilinear_attn_start = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions, hidden_dim*num_layers*self.num_directions)
        
        self.bilinear_attn_end = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions,hidden_dim*num_layers*self.num_directions)
        
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
        
        ctx_embed = self.emb_layer(context)
        # [bs, len_c, emb_dim]
        
        qst_embed = self.emb_layer(question)
        # [bs, len_q, emb_dim]

        align_embed = self.align_embedding(ctx_embed, qst_embed, question_mask)
        # [bs, len_c, emb_dim]  
        
        ctx_bilstm_input = torch.cat([ctx_embed, align_embed], dim=2)
        # [bs, len_c, emb_dim*2]
                
        ctx_encoded = self.context_bilstm(ctx_bilstm_input, context_lengths)
        # [bs, len_c, hid_dim*layers*dir] = [bs, len_c, hid_dim*6]

        qst_encoded = self.question_encoding(qst_embed, question_mask, question_lengths)
        # qtn_weighted = [bs, hid_dim*6]

        start_scores = self.bilinear_attn_start(ctx_encoded, qst_encoded, context_mask)
        # start_scores = [bs, len_c]
         
        end_scores = self.bilinear_attn_end(ctx_encoded, qst_encoded, context_mask)
        # end_scores = [bs, len_c]
        
      
        return start_scores, end_scores
    
class BertQA(nn.Module):

    def __init__(self, device, dropout = 0.1) :     
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

    def __init__(self, device, hidden_dim, freeze, dropout = 0.1) :      
        super().__init__()

        self.device = device

        self.electra = ElectraModel.from_pretrained(globals.ELECTRA_PRETRAINED)
        self.rnn = nn.LSTM(self.electra.config.hidden_size, hidden_dim, batch_first = True, bidirectional = True)
        self.projection =  nn.Linear(hidden_dim*2, hidden_dim)
        self.token_classifier =  nn.Linear(hidden_dim, 2)

        self.relu = nn.ReLU()
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

        proj = self.projection(lstm_out)
        proj = self.relu(proj)
        # [bs, len_txt, lstm_hidden_dim]

        token_scores = self.token_classifier(proj)  #lstm_out
        # [bs, len_txt, 2]

        start_scores, end_scores = token_scores.split(1, dim=-1)
        # [bs, len_txt, 1], [bs, len_txt, 1]

        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)
        # [bs, len_txt]

        start_scores = start_scores.masked_fill(answer_space_mask == 0, float('-inf'))
        end_scores = end_scores.masked_fill(answer_space_mask == 0, float('-inf'))

        return start_scores, end_scores


class Seq2Seq(nn.Module):

    def __init__(self, output_dim, device) :
        super().__init__()

        self.device = device
        self.output_dim = output_dim

        self.to(device)
    
    def get_model_name(self) -> str :
        raise NotImplementedError()
    
    def get_att_mask(self, inputs):
        raise NotImplementedError()
    
    def forward(self, inputs, teacher_force_ratio = 0.5):

        qst_ids = inputs['question_ids']
        mask = self.get_att_mask(inputs)

        batch_size = qst_ids.shape[0]
        trg_len = qst_ids.shape[1]

        #tensor where to store predictions 
        outputs = torch.zeros(batch_size, trg_len, self.output_dim, device=self.device)

        enc_outputs, hidden = self.encoder(inputs)

        input = qst_ids[:,0]    

        for word_idx in range(1,trg_len):

            dec_out, hidden = self.decoder(input, hidden, enc_outputs, mask)

            outputs[:,word_idx,:] = dec_out

            teacher_force = random.random() < teacher_force_ratio 

            pred = dec_out.argmax(dim=1)

            input = qst_ids[:,word_idx] if teacher_force else pred
        
        return outputs

class BertQG(Seq2Seq):

    def __init__(self, dec_vectors, dec_hidden_dim, output_dim, pad_idx, dropout, device) :
        super().__init__(output_dim, device)

        self.encoder = layer.BertEncoder(dropout, device)
        self.decoder = layer.Decoder(dec_vectors, self.encoder.get_hidden_dim(), dec_hidden_dim, output_dim, pad_idx, dropout, device)

        
    
    def get_model_name(self) -> str :
        return 'BertQG'
    
    def get_att_mask(self, inputs):
        return  ~inputs['special_tokens_mask']  # & inputs['type_ids']  #TODO solo ctx o anche answ ? 

class RefNetQG(Seq2Seq):

    def __init__(self, enc_vectors, dec_vectors, enc_hidden_dim, dec_hidden_dim, output_dim, pad_idx, dropout, device) :
        super().__init__(output_dim, device)

        self.encoder = layer.RefNetEncoder(enc_vectors, enc_hidden_dim, dec_hidden_dim, pad_idx, dropout, device)
        self.decoder = layer.Decoder(dec_vectors, self.encoder.get_hidden_dim(), dec_hidden_dim, output_dim, pad_idx, dropout, device )

        
    
    def get_model_name(self) -> str :
        return 'RefNetQG'
    
    def get_att_mask(self, inputs):
        return inputs['context_mask']
    
    

