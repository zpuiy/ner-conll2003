import torch
import torch.nn as nn
import datasets
from conlleval import evaluate
import itertools
from collections import Counter
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence

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

vocab,embeddings = [],[]
embeddings_dict = {}
word2idx_glove = {}
index = 2
with open('glove.6B.100d.txt','rt') as f:
    full_content = f.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    embeddings_dict[i_word] = i_embeddings
    word2idx_glove[i_word] = index
    index += 1
    vocab.append(i_word)
    embeddings.append(i_embeddings)


vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)
word2idx_glove['<pad>'] = 0
word2idx_glove['<unk>'] = 1

n_char_feature = 3

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
char_embs_npa = np.hstack((embs_npa, np.ones((len(embs_npa), 1)), np.zeros((len(embs_npa), 2)))) # all lowercase
case_features = []
unique_words = []
for word, idx in word2idx.items():
    if idx <= 1 or word.islower() or word in word2idx_glove:
        continue
    word_lower = word.lower()
    if word_lower in embeddings_dict:
        case_feature = np.array([0, 0, 0])
        if word.isupper(): # all upper case
            case_feature[1] = 1  # [0, 1, 0]
        elif word.istitle(): # start with capital
            case_feature[2] = 1  #[0, 0, 1]

        case_features.append(case_feature)
        unique_words.append(word)
        word2idx_glove[word] = index
        index += 1

chosen_embs = [embeddings_dict.get(w.lower()) for w in unique_words]
char_embs_npa = np.vstack((char_embs_npa, np.hstack([chosen_embs, case_features]))) 

# use glove vocab to 
dataset_lower = (
    dataset_ori.map(lambda x: {
            'input_ids': [
                word2idx_glove.get(word.lower(), word2idx_glove['<unk>'])
                for word in x['tokens']
            ]
        },
         remove_columns=['id', 'pos_tags', 'chunk_tags']
    )
)

dataset_lower = dataset_lower.rename_column('ner_tags', 'labels')

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
    len_tensors = [len(x) for x in tensors]
    len_targets = [len(y) for y in targets]
    # padding
    features = pad_sequence(tensors, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=9)
    # return features, targets
    return features, targets, len_tensors, len_targets

# dataset['train']['labels'][:2]

trainset = BiLSTM_dataloader(dataset_lower['train']['input_ids'], dataset_lower['train']['labels'])
validset = BiLSTM_dataloader(dataset_lower['validation']['input_ids'], dataset_lower['validation']['labels'])
testset = BiLSTM_dataloader(dataset_lower['test']['input_ids'], dataset_lower['test']['labels'])

batch_size = 32
trainloader = DataLoader(trainset, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)
validloader = DataLoader(validset, collate_fn=collate, batch_size=1, drop_last=True, shuffle=False)
testloader = DataLoader(testset, collate_fn=collate, batch_size=1, drop_last=True, shuffle=False)

#  use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM_GloVe(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout, lstm_layers, num_tags, embs_matrix):
        super(BiLSTM_GloVe, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_matrix).float())
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU(alpha=0.01)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_tags)

    def forward(self, x, labels=None):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        out = self.dropout(out)
        out = self.fc(out)
        out = self.elu(out)
        out = self.classifier(out)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(out.permute(0, 2, 1), labels, ignore_index=9)

        return out, loss
    
# parameters
embedding_dim = 100
lstm_layers = 1
hidden_dim = 256
dropout = 0.33
output_dim = 128

BiLSTM_GloVe_model = BiLSTM_GloVe(
    embedding_dim=embedding_dim + n_char_feature,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    dropout=dropout,
    lstm_layers=lstm_layers,
    num_tags=9,
    embs_matrix=char_embs_npa
).to(device)



BiLSTM_GloVe_model.load_state_dict(torch.load('BiLSTM_GloVe_model.pt'))

# evaluate on the validation set
valid_pred = []

BiLSTM_GloVe_model.eval()
with torch.no_grad():
  for data, label, len_data, len_label in validloader:
      pred, loss = BiLSTM_GloVe_model(data.to(device))

      pred = pred.cpu()
      pred = pred.detach().numpy()
      label = label.detach().numpy()
      pred = np.argmax(pred, axis=2)
      pred = pred.reshape((len(label), -1))[0]
      valid_pred.append(pred.tolist())

# transform labels into NER tags
valid_true = [
  list(map(idx2tag.get, labels))
  for labels in dataset_lower['validation']['labels']
]


valid_pred = [
    list(map(idx2tag.get, pred))
    for pred in valid_pred
]

print("Evaluation on the validation set for task 2:")
precision, recall, f1 = evaluate(
  itertools.chain(*valid_true),
  itertools.chain(*valid_pred))
