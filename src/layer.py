
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AlignQuestionEmbedding(nn.Module):
    
    def __init__(self, input_dim):        
        
        super().__init__()
        
        self.linear = nn.Linear(input_dim, input_dim)  # input_dim = EMBEDDING DIMENSION
        
        self.relu = nn.ReLU()
        
    def forward(self, context, question, question_mask):
        
        # context = [bs, ctx_len, emb_dim]
        # question = [bs, qtn_len, emb_dim]
        # question_mask = [bs, qtn_len]
    
        ctx_ = self.linear(context)
        ctx_ = self.relu(ctx_)
        # ctx_ = [bs, ctx_len, emb_dim]
        
        qtn_ = self.linear(question)
        qtn_ = self.relu(qtn_)
        # qtn_ = [bs, qtn_len, emb_dim]
        
        qtn_transpose = qtn_.permute(0,2,1)
        # qtn_transpose = [bs, emb_dim, qtn_len]
        
        align_scores = torch.bmm(ctx_, qtn_transpose)
        # align_scores = [bs, ctx_len, qtn_len]
        
        qtn_mask = question_mask.unsqueeze(1).expand(align_scores.size())
        # qtn_mask = [bs, 1, qtn_len] => [bs, ctx_len, qtn_len]
        
        # Fills elements of tensor with float(-inf) where mask is 0 (paddding positions). 
        align_scores = align_scores.masked_fill(qtn_mask == 0, float('-inf'))     
        # align_scores = [bs, ctx_len, qtn_len]
        
        align_scores_flat = align_scores.view(-1, question.size(1))
        # align_scores = [bs*ctx_len, qtn_len]
        
        alpha = F.softmax(align_scores_flat, dim=1)
        alpha = alpha.view(-1, context.shape[1], question.shape[1])
        # alpha = [bs, ctx_len, qtn_len]
 
        align_embedding = torch.bmm(alpha, question)
        # align = [bs, ctx_len, emb_dim]
        
        return align_embedding

class StackedBiLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        
        super().__init__()
        
        self.dropout = dropout
        
        self.num_layers = num_layers
        
        self.lstms = nn.ModuleList()
        
        for i in range(self.num_layers):
            
            input_dim = input_dim if i == 0 else hidden_dim * 2
            
            self.lstms.append(nn.LSTM(input_dim, hidden_dim,
                                      batch_first=True, bidirectional=True))
           
    
    def forward(self, x, lengths):
        # x = [bs, seq_len, feature_dim]

        outputs = [x]
        for i in range(self.num_layers):

            lstm_input = outputs[-1]
            lstm_input = F.dropout(lstm_input, p=self.dropout)
            lstm_input_packed = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstms[i](lstm_input_packed)
            lstm_out_padded, out_lengths = pad_packed_sequence(lstm_out, batch_first=True) # [bs, seq_len, hidden_dim * 2]
           
            outputs.append(lstm_out_padded)

    
        output = torch.cat(outputs[1:], dim=2)
        # [bs, seq_len, num_layers*num_dir*hidden_dim]
        
        output = F.dropout(output, p=self.dropout)
      
        return output

class LinearAttentionLayer(nn.Module):
    
    def __init__(self, input_dim):
        
        super().__init__()
        
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, question, question_mask):
        
        # question = [bs, qtn_len, input_dim] = [bs, qtn_len, bi_lstm_hid_dim]
        # question_mask = [bs,  qtn_len]
        
        qtn = question.view(-1, question.shape[-1])
        # qtn = [bs*qtn_len, hid_dim]
        
        attn_scores = self.linear(qtn)
        # attn_scores = [bs*qtn_len, 1]
        
        attn_scores = attn_scores.view(question.shape[0], question.shape[1])
        # attn_scores = [bs, qtn_len]
        
        attn_scores = attn_scores.masked_fill(question_mask == 0, -float('inf'))
        
        alpha = F.softmax(attn_scores, dim=1)
        # alpha = [bs, qtn_len]
        
        return alpha

class QuestionEncodingLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.embedding_size = embedding_dim
        self.hidden_size = hidden_dim
        self.n_layers = num_layers
        self.stacked_bilstms_layer = StackedBiLSTM(input_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(embedding_dim, 1)
    
    def linear_self_attention(self, qst_embed, qst_mask):
        
        scores = self.linear(qst_embed).squeeze(-1) 
        #scores = [batch_size, qst_len]

        scores = scores.masked_fill(qst_mask == 0, float('-inf'))

        return F.softmax(scores, dim=-1)
    
    def forward(self, qst_embed, qst_mask, qst_lengths):
        
        attention_weights = self.linear_self_attention(qst_embed, qst_mask) 
        # attention_weights = [batch_size, qst_len]

        lstm_outputs = self.stacked_bilstms_layer(qst_embed, qst_lengths)
        # lstm_outputs: [batch_size, qst_len, hidden_size * n_layers * 2]

        return torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)

class BilinearAttentionLayer(nn.Module):
    
    def __init__(self, context_dim, question_dim):
        
        super().__init__()
        
        self.linear = nn.Linear(question_dim, context_dim)
        
    def forward(self, context, question, context_mask):
        
        # context = [bs, ctx_len, ctx_hid_dim] = [bs, ctx_len, hid_dim*6] = [bs, ctx_len, 768]
        # question = [bs, qtn_hid_dim] 
        # context_mask = [bs, ctx_len]
        
        qtn_proj = self.linear(question)
        # qtn_proj = [bs, ctx_hid_dim]
        
        qtn_proj = qtn_proj.unsqueeze(2)
        # qtn_proj = [bs, ctx_hid_dim, 1]
        
        scores = context.bmm(qtn_proj)
        # scores = [bs, ctx_len, 1]
        
        scores = scores.squeeze(2)
        # scores = [bs, ctx_len]
        
        scores = scores.masked_fill(context_mask == 0, -float('inf'))
        
        return scores