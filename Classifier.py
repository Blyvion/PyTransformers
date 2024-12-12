#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score


# In[2]:


BATCH_SIZE = 32
MAX_SEQ_LEN = 256
NUM_EPOCHS = 500
NUM_OUTPUTS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ### Data

# In[3]:


ds = load_dataset("Yelp/yelp_review_full")
ds


# In[4]:


def collate_fn(batch):
	# batch is a list of dictionaries
	text = [record['text'] for record in batch]
	label = [record['label'] for record in batch]

	inputs = tokenizer(text,
						  padding = "max_length",
						  truncation = True,
						  max_length = MAX_SEQ_LEN,
						  return_tensors = "pt")

	return {'input_ids': inputs['input_ids'], 'attention_mask': torch.log(inputs['attention_mask']), 'label': torch.tensor(label)}


# In[5]:


SPLITS = 100
train_dataloader = DataLoader(ds['train'].shard(SPLITS, 0), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(ds['test'].shard(SPLITS, 0), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# ### Encoder Only Transformer

# In[6]:


# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens in the sequence.
		The positional encodings have the same dimension as the embeddings, so that the two can be summed.
		Here, we use sine and cosine functions of different frequencies.
	.. math:
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class Classifier(nn.Module):
	def __init__(self, word_embed_size = 512, att_heads = 8, ff_dim = 2048, enc_stack = 6):
		super(Classifier, self).__init__()
		self.embedding = nn.Embedding(35000, word_embed_size)
		self.positional = PositionalEncoding(word_embed_size)
		self.d_model = word_embed_size

		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model = word_embed_size,
			nhead = att_heads,
			dim_feedforward = ff_dim,
			batch_first = True
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, enc_stack)

		# classifier
		self.linear1 = nn.Linear(word_embed_size, NUM_OUTPUTS)
	
	def forward(self, input, src_mask = None):
		out = self.embedding(input) * math.sqrt(self.d_model)
		out = self.positional(out)
		
		out = self.encoder(out, src_key_padding_mask = src_mask)
		out = torch.mean(out, dim=1)
		# out = out[:, 0, :].squeeze(1)
		out = torch.softmax(self.linear1(out), dim=-1)

		return out


# ### Training

# In[7]:


model = Classifier().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5) # learning rate need to be low enough


# In[8]:


losses = list()
accuracy = list()

for epoch in range(NUM_EPOCHS):

	with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch', ncols=110) as pbar:

		running_loss = 0.0
		running_acc = 0.0

		for i, batch in enumerate(train_dataloader):
			input_ids = batch['input_ids'].to(device)
			mask = batch['attention_mask'].to(device)
			label = batch['label'].to(device)

			optimizer.zero_grad()

			outputs = model(input_ids, mask)
			loss = loss_fn(outputs, label)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, pred = outputs.topk(1, dim=1)
			running_acc += accuracy_score(label.cpu(), pred.cpu().flatten())
			pbar.set_postfix(loss=f'{running_loss/((i+1)):.4f}',
					accuracy=f'{running_acc/((i+1)):.4f}')
			pbar.update()
		
		losses.append(running_loss/len(train_dataloader))
		accuracy.append(running_acc/len(train_dataloader))

	with tqdm(total=len(test_dataloader), desc=f'Test', unit='batch', ncols=90) as pbar:

		test_running_loss = 0.0
		test_running_acc = 0.0

		with torch.no_grad():
			for batch in test_dataloader:
				input_ids = batch['input_ids'].to(device)
				mask = batch['attention_mask'].to(device)
				label = batch['label'].to(device)

				outputs = model(input_ids, mask)
				loss = loss_fn(outputs, label)
				running_loss += loss.item()
				_, pred = outputs.topk(1, dim=1)
				running_acc += accuracy_score(label.cpu(), pred.cpu().flatten())
				pbar.set_postfix(loss=f'{running_loss/((i+1)):.4f}',
					accuracy=f'{running_acc/((i+1)):.4f}')
				pbar.update()
	
print('\nFinished Training')


# In[ ]:


torch.save(model.state_dict(), "model_weights.pth")


# #### Resources
# https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/
