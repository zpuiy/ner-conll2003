import torch
import torch.nn as nn
import datasets
from conlleval import evaluate
import itertools
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
import math
from torch import Tensor

dataset_ori = datasets.load_dataset("conll2003")

word_frequency = Counter(itertools.chain(*dataset_ori['train']['tokens']))  # type: ignore

# Remove words below threshold 3
word_frequency = {
    word: frequency
    for word, frequency in word_frequency.items()
    if frequency >= 3
}

word2idx = {
    word: index
    for index, word in enumerate(word_frequency.keys(), start=2)
}

word2idx['[PAD]'] = 0
word2idx['[UNK]'] = 1

idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}


dataset = (
    dataset_ori.map(lambda x: {
            'input_ids': [
                word2idx.get(word, word2idx['[UNK]'])
                for word in x['tokens']
            ]
        },
         remove_columns=['id', 'pos_tags', 'chunk_tags']
    )
)

dataset = dataset.rename_column('ner_tags', 'labels')

# Define custom dataset class for train set
class BiLSTM_dataloader(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __getitem__(self,idx):
    return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

  def __len__(self):
    return len(self.x)

def collate(data):
    tensors, targets = zip(*data)
    # padding
    features = pad_sequence(tensors, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=9)
    return features, targets


trainset = BiLSTM_dataloader(dataset['train']['input_ids'], dataset['train']['labels'])
validset = BiLSTM_dataloader(dataset['validation']['input_ids'], dataset['validation']['labels'])
testset = BiLSTM_dataloader(dataset['test']['input_ids'], dataset['test']['labels'])

# parameters
embedding_dim = 128
num_heads = 8
max_length = 128
ff_dim = 128
dropout = 0.1


#  use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# position embedding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_dim, 2)* math.log(10000) / embedding_dim)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, embedding_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        # return (token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# token embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)

def create_mask(src, device):
    src_seq_len = src.shape[0]
    # tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == 0).transpose(0, 1)
    # tgt_padding_mask = (tgt == 9).transpose(0, 1)
    return src_mask, src_padding_mask

# transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, ff_dim, num_tags, max_length, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_length)
        self.src_tok_emb = TokenEmbedding(vocab_size, embedding_dim)
        # self.tgt_tok_emb = TokenEmbedding(num_tags, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) 
        self.classifier = nn.Linear(embedding_dim, num_tags)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, src_key_padding_mask, labels=None):
        src_emb = self.positional_encoding(self.src_tok_emb(x))
        # tgt_emb = self.positional_encoding(self.tgt_tok_emb(labels))
        out = self.transformer_encoder(src_emb, src_mask, src_key_padding_mask)
        out = self.dropout(out)
        # out = self.layer_norm(out)
        out = self.classifier(out)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(out.permute(0, 2, 1), labels, ignore_index=9)
            # loss = nn.functional.cross_entropy(out, labels, ignore_index=9)

        return out, loss
    

Transformer_model = TransformerModel(vocab_size=len(word2idx), 
                                     embedding_dim=embedding_dim, 
                                     n_heads=num_heads, 
                                     ff_dim=ff_dim, 
                                     num_tags=len(idx2tag), 
                                     max_length=max_length,
                                     dropout=dropout,
                                     ).to(device)

Transformer_model.load_state_dict(torch.load('Transformer_model.pt'))

#DataLoader
batch_size = 4
trainloader = DataLoader(trainset, collate_fn=collate, batch_size=batch_size, shuffle=True)
validloader = DataLoader(validset, collate_fn=collate, batch_size=1, shuffle=False)
testloader = DataLoader(testset, collate_fn=collate, batch_size=1, shuffle=False)

# evaluate on the validation set
valid_pred = []

Transformer_model.eval()
with torch.no_grad():
  for data, label in validloader:
      data = data.to(device)
      src_mask, src_padding_mask = create_mask(data, device)
      pred, loss = Transformer_model(data, src_mask, src_padding_mask)

      pred = pred.cpu()
      pred = pred.detach().numpy()
      label = label.detach().numpy()
      pred = np.argmax(pred, axis=2)
      pred = pred.reshape((len(label), -1))[0]
      valid_pred.append(pred.tolist())

# transform labels into NER tags
valid_true = [
  list(map(idx2tag.get, labels))
  for labels in dataset['validation']['labels']
]


valid_pred = [
    list(map(idx2tag.get, pred))
    for pred in valid_pred
]

print("Evaluation on the validation set:")
precision, recall, f1 = evaluate(
  itertools.chain(*valid_true),
  itertools.chain(*valid_pred))
