from turtle import forward
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingLayer(nn.Module):

    def __init__(self, vectors : np.ndarray, pad_idx : int, hook, device = 'cpu'):
        super().__init__()

        vectors = torch.from_numpy(vectors).to(device) 
        
        _ , self.embedding_dim = vectors.shape
        self.embed = nn.Embedding.from_pretrained(vectors, freeze = False, padding_idx = pad_idx)   #load pretrained weights in the layer 

        if hook is not None : self.embed.weight.register_hook(hook) 
    
    def forward(self, to_embed):

        embeddings = self.embed(to_embed)
        return embeddings


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
        
        attn_scores = attn_scores.masked_fill(question_mask == 0, float('-inf'))
        
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
        
        scores = scores.masked_fill(context_mask == 0, float('-inf'))
        
        return scores


class Encoder(nn.Module):

    def __init__(self, vectors, enc_hidden_dim, dec_hidden_dim, pad_idx, device):
        super().__init__()

        self.device = device
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.emb_layer = EmbeddingLayer(vectors, pad_idx, None, device)
        self.emb_dim = self.emb_layer.embedding_dim

        self.ctx_rnn = nn.LSTM(self.emb_dim+1, enc_hidden_dim, batch_first=True, bidirectional=True)
        self.answ_rnn = nn.LSTM((self.enc_hidden_dim*2)+self.emb_dim, enc_hidden_dim, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(enc_hidden_dim*2, enc_hidden_dim*2)  #TODO cambiare nomi 
        self.fc2 = nn.Linear(enc_hidden_dim*2, dec_hidden_dim)

        self.tanh = nn.Tanh()

        #TODO dropout 
    
    def augment_embeddings(self,embeddings,answ_start,answ_end):

        t1 = torch.le(answ_start.unsqueeze(-1),torch.arange(embeddings.shape[1],device=self.device)).float()   #TODO rename e spiegare cosa facciamo 
        t2 = torch.ge(answ_end.unsqueeze(-1),torch.arange(embeddings.shape[1])).float()
        m = torch.mul(t1,t2).unsqueeze(-1)
        augmented_emb = torch.cat((embeddings,m),dim=2)

        return augmented_emb
    
    def ctx2answ(self,answ_embeds,ctx_out,answ_start,answ_end):

        z = torch.zeros(answ_embeds.shape[0],answ_embeds.shape[1],self.enc_hidden_dim*2)    #TODO rename

        for i in range(answ_embeds.shape[0]):
            z[i,0:answ_end[i]+1-answ_start[i],:] = ctx_out[i,answ_start[i]:answ_end[i]+1,:]   #TODO no for loop

        return torch.cat((z,answ_embeds),dim=2)

    
    def forward(self, context_ids, answer_ids, answ_start, answ_end):

        #context_ids = [bs, ctx_len]
        #answ_start = [bs]
        #answ_end = [bs]

        ctx_embeds = self.emb_layer(context_ids)
        # [bs, ctx_len, emb_dim]
        
        augmented_emb = self.augment_embeddings(ctx_embeds,answ_start,answ_end)
        # [bs, ctx_len, emb_dim+1]

        ctx_outputs, (ctx_hidden,ctx_cell) = self.ctx_rnn(augmented_emb)    #TODO pack padded ecc 
        # [bs, ctx_len, enc_hidden_dim*2]

        answ_embeds = self.emb_layer(answer_ids)
        # [bs, answ_len, emb_dim]

        answ_ctx = self.ctx2answ(answ_embeds,ctx_outputs,answ_start,answ_end)
        # [bs, answ_len, enc_hidden_dim*2 + emb_dim]

        answ_outputs, (answ_hidden,answ_cell) = self.answ_rnn(answ_ctx)
        # [bs, anw_len, enc_hidden_dim*2]

        answ_reduced = torch.mean(answ_outputs,dim=1) # [bs, enc_hidden_dim*2]
        ctx_reduced = torch.mean(ctx_outputs,dim=1) # [bs, enc_hidden_dim*2]              

        projected_answ = self.fc1(answ_reduced)  
        # [bs, enc_hidden_dim*2]

        ctx_answ_fusion = torch.add(projected_answ,ctx_reduced)
        # [bs, enc_hidden_dim*2]

        hidden = self.tanh(self.fc2(ctx_answ_fusion))
        # [bs, dec_hidden_dim]

        return ctx_outputs, hidden


class Attention(nn.Module):

    def __init__(self, dec_hidden_dim, enc_hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(dec_hidden_dim + (enc_hidden_dim*2), dec_hidden_dim) #TODO rename 
        self.fc2 = nn.Linear(dec_hidden_dim, 1, bias=False)   #TODO bias 

        self.tanh = nn.Tanh()


    def forward(self, hidden, outputs, ):

        #outputs = [bs, ctx_len, enc_hidden_dim*2]
        #hidden = [bs, dec_hidden_dim]

        hidden = hidden.unsqueeze(1).repeat(1,outputs.shape[1],1)
        # [bs, ctx_len, dec_hidden_dim]

        temp1 = self.fc1(torch.cat((hidden,outputs), dim=2))   #TODO RENAME TEMP 
        temp2 = self.tanh(temp1)
        # [bs, ctx_len, dec_hidden_dim]

        a = self.fc2(temp2).squeeze(2)
        # [bs, ctx_len]

        #TODO MASK FILL 

        soft_a = F.softmax(a, dim=1)
        # [bs, ctx_len]

        weighted_outputs = torch.bmm(soft_a.unsqueeze(1),outputs)
        # [bs, 1, enc_hidden_dim*2]

        return weighted_outputs


class Decoder(nn.Module):

    def __init__(self, vectors, enc_hidden_dim, dec_hidden_dim, dec_output_dim, pad_idx, device):
        super().__init__()

        self.emb_layer = EmbeddingLayer(vectors, pad_idx, None, device)
        self.emb_dim = self.emb_layer.embedding_dim

        self.attention = Attention(dec_hidden_dim, enc_hidden_dim)

        self.rnn = nn.LSTM((enc_hidden_dim*2)+self.emb_dim, dec_hidden_dim, batch_first=True)

        self.fc_out = nn.Linear((enc_hidden_dim*2)+dec_hidden_dim, dec_output_dim)   #TODO name 

    
    def forward(self, input, prev_hidden, prev_cell, enc_outputs):

        #input = [bs]
        #prev_hidden = [bs, dec_hidden_dim]
        #prev_cell = [bs, dec_hidden_dim]
        #enc_outputs = [bs, ctx_len, enc_hidden_dim*2]

        input = input.unsqueeze(1)
        # [bs, 1]

        qst_embeds = self.emb_layer(input)
        # [bs, 1, emb_dim]

        ctx_vector = self.attention(prev_hidden, enc_outputs)   #TODO rename
        # [bs, 1, enc_hidden_dim*2]

        rnn_input_0 = torch.cat((qst_embeds,ctx_vector), dim=2)
        # [bs, 1, (enc_hidden_dim*2) + emb_dim]

        rnn_input_1 = prev_hidden.unsqueeze(0), prev_cell.unsqueeze(0)

        rnn_out , (rnn_hidden, rnn_cell) = self.rnn(rnn_input_0,rnn_input_1)
        # rnn_out = [bs, 1, dec_hidden_dim]

        rnn_out = rnn_out.squeeze(1)          #[bs, dec_hidden_dim]
        ctx_vector = ctx_vector.squeeze(1)    #[bs, enc_hidden_dim*2]
        rnn_hidden = rnn_hidden.squeeze(0)
        rnn_cell = rnn_cell.squeeze(0)

        dec_out = self.fc_out(torch.cat((rnn_out,ctx_vector), dim=1))
        # [bs, dec_output_dim]

        return dec_out, rnn_hidden, rnn_cell












