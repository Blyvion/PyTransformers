#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import math
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# In[ ]:


BATCH_SIZE = 2
MAX_SEQ_LEN = 256
NUM_EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### Data

# In[ ]:


class YelpDataset(Dataset):
	def __init__(self, data, tokenizer, max_length = None):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = 512

	def __len__(self):
		return self.data.num_rows
	
	def __getitem__(self, idx):
		text = self.data['text'][idx]
		label = self.data['label'][idx]

		inputs = self.tokenizer(text,
						  padding = "max_length",
						  truncation = True,
						  max_length = self.max_length,
						  return_tensors = "pt")
		
		input_ids = inputs['input_ids'].squeeze()
		attention_mask = inputs['attention_mask'].squeeze() # mask pads.
		attention_mask = torch.where(attention_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf')))
		
		return {"input_ids": input_ids, "attention_mask": attention_mask}, label


# In[ ]:


ds = load_dataset("Yelp/yelp_review_full")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_dataset = YelpDataset(ds['train'], tokenizer)
test_dataset = YelpDataset(ds['test'], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ### Encoder Only Transformer

# In[ ]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) to store the positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector with shape (max_len, 1) containing the positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create a row vector with shape (1, d_model//2) containing the frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute the positional encodings and store them in the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register the positional encodings as a buffer, which means it won't be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input tensor
        # x has shape (batch_size, seq_len, d_model)
        # pe[:, :x.size(1), :] selects the positional encodings up to the length of the input sequence
        x = x + self.pe[:, :x.size(1), :]
        return x


# In[ ]:


class Classifier(nn.Module):
	def __init__(self, word_embed_size = 800, att_heads = 10, ff_dim = 2048, enc_stack = 8):
		super(Classifier, self).__init__()
		self.embedding = nn.Embedding(35000, word_embed_size)
		self.positional = PositionalEncoding(word_embed_size)

		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model = word_embed_size,
			nhead = att_heads,
			dim_feedforward = ff_dim,
			batch_first = True
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, enc_stack)

		# classifier
		self.linear1 = nn.Linear(word_embed_size, word_embed_size)
		self.linear2 = nn.Linear(word_embed_size, 5)
	
	def forward(self, input, src_mask = None):
		out = self.embedding(input)
		out = self.positional(out)
		
		out = self.encoder(out, src_key_padding_mask = src_mask)
		out = torch.mean(out, dim = 1)
		out = torch.relu(self.linear1(out))
		out = torch.softmax(self.linear2(out), dim = -1)

		return out


# ### Training

# In[ ]:


model = Classifier().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


for epoch in range(NUM_EPOCHS):

	running_loss = 0.0
	with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch', ncols=100) as pbar:
		for data, label in train_dataloader:
			input_ids = data['input_ids'].to(device)
			mask = data['attention_mask'].to(device)
			label = label.to(device)

			optimizer.zero_grad()

			outputs = model(input_ids, mask)
			loss = loss_fn(outputs, label)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			pbar.set_postfix(loss=f'{running_loss/len(train_dataloader):.4f}')
			pbar.update(1)
	
	#print(f'\r {running_loss/len(dataloader)}', end='', flush=True)

print('\nFinished Training')

