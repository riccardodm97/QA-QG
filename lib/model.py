import torch
from torch import nn 

import lib.layer as layer 
from lib.utils import get_embedding_layer

class DrQA(nn.Module):
    
    def __init__(self, hidden_dim, num_layers, dropout, weights_matrix, pad_idx, device):
        
        super().__init__()
        
        self.device = device      
        self.num_directions = 2   #bidirectional LSTMs
        
        self.embedding_layer, self.embedding_dim = get_embedding_layer(weights_matrix, pad_idx, device)
        
        self.context_bilstm = layer.StackedBiLSTM(self.embedding_dim * 2, hidden_dim, num_layers, dropout)
        
        self.question_bilstm = layer.StackedBiLSTM(self.embedding_dim, hidden_dim, num_layers, dropout)
        
        def tune_embedding(grad, words=1000):
            grad[words:] = 0
            return grad
        
        self.embedding_layer.weight.register_hook(tune_embedding)
        
        self.align_embedding = layer.AlignQuestionEmbedding(self.embedding_dim)
        
        self.linear_attn_question = layer.LinearAttentionLayer(hidden_dim*num_layers*self.num_directions) 
        
        self.bilinear_attn_start = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions, 
                                                          hidden_dim*num_layers*self.num_directions)
        
        self.bilinear_attn_end = layer.BilinearAttentionLayer(hidden_dim*num_layers*self.num_directions,
                                                        hidden_dim*num_layers*self.num_directions)
        
        self.dropout = nn.Dropout(dropout)

        self.to(self.device)
      
    
    def forward(self, inputs):   
       
        context = inputs['context_ids']              # [bs, len_c]
        question = inputs['question_ids']            # [bs, len_q]
        context_mask = inputs['context_mask']        # [bs, len_c]
        question_mask = inputs['question_mask']      # [bs, len_q]
        
        ctx_embed = self.embedding_layer(context)
        # ctx_embed = [bs, len_c, emb_dim]
        
        ques_embed = self.embedding_layer(question)
        # ques_embed = [bs, len_q, emb_dim]

        ctx_embed = self.dropout(ctx_embed)
     
        ques_embed = self.dropout(ques_embed)
             
        align_embed = self.align_embedding(ctx_embed, ques_embed, question_mask)
        # align_embed = [bs, len_c, emb_dim]  
        
        ctx_bilstm_input = torch.cat([ctx_embed, align_embed], dim=2)
        # ctx_bilstm_input = [bs, len_c, emb_dim*2]
                
        ctx_outputs = self.context_bilstm(ctx_bilstm_input)
        # ctx_outputs = [bs, len_c, hid_dim*layers*dir] = [bs, len_c, hid_dim*6]
       
        qtn_outputs = self.question_bilstm(ques_embed)
        # qtn_outputs = [bs, len_q, hid_dim*6]
    
        qtn_weights = self.linear_attn_question(qtn_outputs, question_mask)
        # qtn_weights = [bs, len_q]
            
        qtn_weighted = layer.LinearAttentionLayer.weighted_average(qtn_outputs, qtn_weights)
        # qtn_weighted = [bs, hid_dim*6]
        
        start_scores = self.bilinear_attn_start(ctx_outputs, qtn_weighted, context_mask)
        # start_scores = [bs, len_c]
         
        end_scores = self.bilinear_attn_end(ctx_outputs, qtn_weighted, context_mask)
        # end_scores = [bs, len_c]
        
      
        return start_scores, end_scores