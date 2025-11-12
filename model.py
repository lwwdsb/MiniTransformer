import torch
import torch.nn as nn
import math
from vocab import PAD

def create_subsequent_mask(size: int):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()           # initialize parent class
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]# Add positional encoding,x.size(1) is seq_len,not max_len
        return x
      
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads# dimension per head
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.size()  # Batch size, Src_len, D_model
        _, S, _ = key.size()   # Batch size, Src_len, D_model
        
        # Calculate query, key, value
        # B, L, d_model -> B, num_heads, L, d_k,transpose is to swap L and num_heads to facilitate matmul for each head
        Q = self.w_q(query).view(B, L, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(key).view(B, S, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(value).view(B, S, self.num_heads, self.d_k).transpose(1,2)
        # transpose to make matmul for Q and KT
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn.matmul(V)
        # Concat heads
        out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        out = self.fc(out)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
      
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        att_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(att_out))
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))
        return src
      
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Self attention
        attn1 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn1))
        # Cross attention
        attn2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn2))
        
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt
      
class MiniTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=256, max_len=100):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout = 0.1) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout = 0.1) for _ in range(num_layers)])
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
    def make_src_mask(self, src):
        src_mask = (src == PAD).unsqueeze(1).unsqueeze(2)
        return src_mask  # [B, 1, 1, src_len]
      
    def make_tgt_mask(self, tgt):
        B, L = tgt.size()
        pad_mask = (tgt == PAD).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        subseq_mask = create_subsequent_mask(L).to(tgt.device).unsqueeze(0).unsqueeze(1)
        tgt_mask = pad_mask | subseq_mask  # 两种mask结合
        return tgt_mask  # [B, 1, L, L]
      
    def forward(self, src, tgt):
    # ---------- 1️⃣ mask ----------
        src_mask = self.make_src_mask(src)                             # [B, 1, 1, src_len]
        tgt_mask = self.make_tgt_mask(tgt)                             # [B, 1, tgt_len, tgt_len]
        memory_mask = src_mask.expand(-1, 1, tgt.size(1), src.size(1)) # [B, 1, tgt_len, src_len]

    # ---------- 2️⃣ encoder ----------
        src_emb = self.positional_encoding(self.src_embedding(src))
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
        memory = src_emb  # [B, src_len, d_model]

    # ---------- 3️⃣ decoder ----------
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory, tgt_mask, memory_mask)

    # ---------- 4️⃣ 输出 ----------
        out = self.generator(tgt_emb)
        return out
        
        
        
        
        
        