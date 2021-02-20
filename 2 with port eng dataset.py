import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from torchtext import data

import spacy
from tqdm import tqdm 
import pandas as pd
import random, os
import math, time

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_pt = spacy.load('pt_core_news_md')
spacy_en = spacy.load('en_core_web_md')

def tokenize_pt(text):
    """
    Tokenizes Port text from a string into a list of strings
    """
    return [tok.text for tok in spacy_pt.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def read_data(src_data, trg_data):
    
    if src_data is not None:
        try:
            src_data = open(src_data).read().strip().split('\n')
        except:
            print("error: '" + src_data + "' file not found")
            quit()
    
    if trg_data is not None:
        try:
            trg_data = open(trg_data).read().strip().split('\n')
        except:
            print("error: '" + trg_data + "' file not found")
            quit()
    return src_data, trg_data

src_train_data_path = '../pt_customized_transformer/data/port_train.txt'
trg_train_data_path = '../pt_customized_transformer/data/eng_train.txt'
src_test_data_path = '../pt_customized_transformer/data/port_test.txt'
trg_test_data_path = '../pt_customized_transformer/data/eng_test.txt'
src_dev_data_path = '../pt_customized_transformer/data/port_dev.txt'
trg_dev_data_path = '../pt_customized_transformer/data/eng_dev.txt'

src_data_train, trg_data_train = read_data(src_train_data_path, trg_train_data_path)
src_data_test, trg_data_test = read_data(src_test_data_path, trg_test_data_path)
src_data_dev, trg_data_dev = read_data(src_dev_data_path, trg_dev_data_path)

SRC = Field(tokenize=tokenize_pt, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

TRG = Field(tokenize = tokenize_en, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

def generate_dataset(src_data, trg_data):

    raw_data = {'src' : [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    data_fields = [('src', SRC), ('trg', TRG)]
    df.to_csv("translate_dataset_temp.csv", index=False)
    dataset = data.TabularDataset('./translate_dataset_temp.csv', format='csv', fields=data_fields)
    os.remove('translate_dataset_temp.csv')
    return dataset

train = generate_dataset(src_data_train, trg_data_train)
test = generate_dataset(src_data_test, trg_data_test)
dev = generate_dataset(src_data_dev, trg_data_dev)

# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
#                                                     fields = (SRC, TRG))

# print(vars(train_data.examples[0]))

SRC.build_vocab(train, min_freq = 2)
TRG.build_vocab(train, min_freq = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train, dev, test), sort_key=lambda x: (len(x.src), len(x.trg)),
     batch_size = BATCH_SIZE, device = device)

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 emb_dim: int, 
                 hid_dim: int, 
                 dropout: float):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
        
        self.rnn = nn.GRU(emb_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim: int, 
                 emb_dim: int, 
                 hid_dim: int, 
                 dropout: float):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        
        self.out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                input: Tensor, 
                hidden: Tensor, 
                context: Tensor):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #sent len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 device: torch.device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, 
                src: Tensor, 
                trg: Tensor, 
                teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        context = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, context) # input and hidden are new, context is fixed
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
        
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i

def train(model: nn.Module, 
          iterator: BucketIterator, 
          optimizer: optim.Adam, 
          criterion: nn.modules.loss.CrossEntropyLoss, 
          clip: float):
    
    model.train()
    
    epoch_loss = 0

    total_len = get_len(iterator)
    print(f'Iterator has {total_len} batches.')
    
    for i, batch in tqdm(enumerate(iterator)):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
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
    
        for i, batch in tqdm(enumerate(iterator)):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time: int, 
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'seq2seq_weights/model_weights')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('seq2seq_weights/model_weights'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')