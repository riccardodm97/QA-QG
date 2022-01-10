import torch
from torch import nn
from layer import *

class DrQA(nn.Module):
    
    def __init__(self, hidden_dim, num_layers, num_directions, dropout, weights_matrix, pad_idx, device):
        
        super().__init__()
        
        self.device = device
        
        #self.embedding = self.get_glove_embedding()
        self.embedding_layer, self.embedding_dim = self.get_glove_embedding(weights_matrix, pad_idx)
        
        self.context_bilstm = StackedBiLSTM(self.embedding_dim * 2, hidden_dim, num_layers, dropout)
        
        self.question_bilstm = StackedBiLSTM(self.embedding_dim, hidden_dim, num_layers, dropout)
        
        def tune_embedding(grad, words=1000):
            grad[words:] = 0
            return grad
        
        self.embedding_layer.weight.register_hook(tune_embedding)
        
        self.align_embedding = AlignQuestionEmbedding(self.embedding_dim)
        
        self.linear_attn_question = LinearAttentionLayer(hidden_dim*num_layers*num_directions) 
        
        self.bilinear_attn_start = BilinearAttentionLayer(hidden_dim*num_layers*num_directions, 
                                                          hidden_dim*num_layers*num_directions)
        
        self.bilinear_attn_end = BilinearAttentionLayer(hidden_dim*num_layers*num_directions,
                                                        hidden_dim*num_layers*num_directions)
        
        self.dropout = nn.Dropout(dropout)
   
        
    def get_glove_embedding(self, weights_matrix, pad_idx):
        
        # TODO: inserire weigths_matrix
        # weights_matrix = np.load('drqaglove_vt.npy')
        _, embedding_dim = weights_matrix.shape
        embedding_layer = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx = pad_idx)   #load pretrained weights in the layer and make it non-trainable

        return embedding_layer, embedding_dim
    
    
    def forward(self, context, question, context_mask, question_mask):
       
        # context = [bs, len_c]
        # question = [bs, len_q]
        # context_mask = [bs, len_c]
        # question_mask = [bs, len_q]
        
        
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
            
        qtn_weighted = LinearAttentionLayer.weighted_average(qtn_outputs, qtn_weights)
        # qtn_weighted = [bs, hid_dim*6]
        
        start_scores = self.bilinear_attn_start(ctx_outputs, qtn_weighted, context_mask)
        # start_scores = [bs, len_c]
         
        end_scores = self.bilinear_attn_end(ctx_outputs, qtn_weighted, context_mask)
        # end_scores = [bs, len_c]
        
      
        return start_scores, end_scores