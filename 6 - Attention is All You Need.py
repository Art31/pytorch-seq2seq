import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os
import time

SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_md')
spacy_en = spacy.load('en_core_web_md')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size=BATCH_SIZE,
     device=device)

class SelfAttention(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 n_heads: int, 
                 dropout: float, 
                 device: torch.device):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        
    def forward(self, 
                query: Tensor, 
                key: Tensor, 
                value: Tensor, 
                mask: Tensor = None):
        
        bsz = query.shape[0]
        
        #query = key = value [batch size, sent len, hid dim]
                
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        #Q, K, V = [batch size, sent len, hid dim]
        
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        #Q, K, V = [batch size, n heads, sent len, hid dim // n heads]
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, sent len, sent len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = self.dropout(F.softmax(energy, dim=-1))
        
        #attention = [batch size, n heads, sent len, sent len]
        
        x = torch.matmul(attention, V)
        
        #x = [batch size, n heads, sent len, hid dim // n heads]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, sent len, n heads, hid dim // n heads]
        
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        
        #x = [batch size, src sent len, hid dim]
        
        x = self.fc(x)
        
        #x = [batch size, sent len, hid dim]
        
        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 pf_dim: int, 
                 dropout: float):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, 
                x: Tensor):
        
        #x = [batch size, sent len, hid dim]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, hid dim, sent len]
        
        x = self.dropout(F.relu(self.fc_1(x)))
        
        #x = [batch size, ff dim, sent len]
        
        x = self.fc_2(x)
        
        #x = [batch size, hid dim, sent len]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, sent len, hid dim]
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 n_heads: int, 
                 pf_dim: int, 
                 self_attention: SelfAttention, 
                 positionwise_feedforward: PositionwiseFeedforward, 
                 dropout: float, 
                 device: torch.device):
        super().__init__()      
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = self_attention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                src: Tensor, 
                src_mask: Tensor):
        
        #src = [batch size, src sent len, hid dim]
        #src_mask = [batch size, 1, 1, src sent len]
 
        #src = [batch size, src sent len, hid dim]
        src = self.layer_norm(
            src + self.dropout(self.self_attention(
                src, src, src, src_mask)))
        
        #src = [batch size, src sent len, hid dim]        
        src = self.layer_norm(
            src + self.dropout(
                self.positionwise_feedforward(src)))
        
        return src

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hid_dim: int, 
                 n_layers: int, 
                 n_heads: int, 
                 pf_dim: int, 
                 encoder_layer: EncoderLayer, 
                 self_attention: SelfAttention, 
                 positionwise_feedforward: PositionwiseFeedforward, 
                 dropout: float, 
                 device: torch.device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, 
                src: Tensor, 
                src_mask: Tensor):      
        #src = [batch size, src sent len]
        #src_mask = [batch size, 1, 1, src sent len]
        
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        #pos = [batch size, src sent len]
        
        src_embedded = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src sent len, hid dim]
        
        # each layer is an "EncoderLayer"
        for layer in self.layers:
            src_embedded = layer(src_embedded, src_mask)
            
        return src_embedded

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 n_heads: int, 
                 pf_dim: int, 
                 self_attention: SelfAttention, 
                 positionwise_feedforward: PositionwiseFeedforward, 
                 dropout: float, 
                 device: torch.device):
        super().__init__()
        
        self.layer_nore = nn.LayerNorm(hid_dim)
        self.self_attention = self_attention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = self_attention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                trg: Tensor, 
                src: Tensor, 
                trg_mask: Tensor, 
                src_mask: Tensor):
        
        #trg = [batch size, trg sent len, hid dim]
        #src = [batch size, src sent len, hid dim]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]
                
        trg = self.layer_norm(
            trg + self.dropout(
                self.self_attention(trg, trg, trg, trg_mask)))
                
        trg = self.layer_norm(
            trg + self.do(
                self.encoder_attention(trg, src, src, src_mask)))
        
        trg = self.layer_norm(
            trg + self.dropout(
                self.positionwise_feedforward(trg)))
        
        return trg

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim: int, 
                 hid_dim: int, 
                 n_layers: int, 
                 n_heads: int, 
                 pf_dim: int, 
                 decoder_layer: DecoderLayer, 
                 self_attention: SelfAttention, 
                 positionwise_feedforward: PositionwiseFeedforward, 
                 dropout: float, 
                 device: torch.device):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, 
                                                   self_attention, 
                                                   positionwise_feedforward, 
                                                   dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, 
                trg: Tensor, 
                src: Tensor, 
                trg_mask: Tensor, 
                src_mask: Tensor):            
        #trg = [batch_size, trg sent len]
        #src = [batch_size, src sent len]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]
        
        #pos = [batch_size, trg sent len]
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        #trg = [batch_size, trg sent len]        
        trg_embedded = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        #trg = [batch size, trg sent len, hid dim]
        
        #trg_mask = [batch size, 1, trg sent len, trg sent len]      
        
        for layer in self.layers:
            trg_embedded = layer(trg_embedded, src, trg_mask, src_mask)

        #trg = [batch size, trg sent len, hid dim]            

        return self.fc(trg_embedded)

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 pad_idx: int, 
                 device: torch.device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        
    def make_masks(self, 
                   src: Tensor, 
                   trg: Tensor):
        
        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]
        
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))
        
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return src_mask, trg_mask
    
    def forward(self, 
                src: Tensor, 
                trg: Tensor):
        
        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]
                
        src_mask, trg_mask = self.make_masks(src, trg)
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src sent len, hid dim]
                
        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #out = [batch size, trg sent len, output dim]
        
        return out

input_dim = len(SRC.vocab)
hid_dim = 512
# n_layers = 2
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1

enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

output_dim = len(TRG.vocab)
hid_dim = 512
n_layers = 6
# n_layers = 2
n_heads = 8
pf_dim = 2048
dropout = 0.1

dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = SRC.vocab.stoi['<pad>']

model = Seq2Seq(enc, dec, pad_idx, device).to(device)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, 
                 model_size: int, 
                 factor: int, 
                 warmup: int, 
                 optimizer: Optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

optimizer = NoamOpt(hid_dim, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model: nn.Module, 
          iterator: BucketIterator, 
          optimizer: optim.Adam, 
          criterion: nn.modules.loss.CrossEntropyLoss, 
          clip: float):

    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.optimizer.zero_grad()
        
        output = model(src, trg[:,:-1])
                
        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]
            
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model: nn.Module, 
             iterator: BucketIterator, 
             criterion: nn.modules.loss.CrossEntropyLoss):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1])
            
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]
            
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time: int, 
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 1
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')