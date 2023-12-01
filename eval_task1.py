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
    len_tensors = [len(x) for x in tensors]
    len_targets = [len(y) for y in targets]
    # padding
    features = pad_sequence(tensors, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=9)
    # return features, targets
    return features, targets, len_tensors, len_targets


trainset = BiLSTM_dataloader(dataset['train']['input_ids'], dataset['train']['labels'])
validset = BiLSTM_dataloader(dataset['validation']['input_ids'], dataset['validation']['labels'])
testset = BiLSTM_dataloader(dataset['test']['input_ids'], dataset['test']['labels'])

#DataLoader
batch_size = 8
trainloader = DataLoader(trainset, collate_fn=collate, batch_size=batch_size, shuffle=True)
validloader = DataLoader(validset, collate_fn=collate, batch_size=1, shuffle=False)
testloader = DataLoader(testset, collate_fn=collate, batch_size=1, shuffle=False)

#  use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, lstm_layers, num_tags):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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

BiLSTM_model = BiLSTM(
    vocab_size=len(word2idx),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    dropout=dropout,
    lstm_layers=lstm_layers,
    num_tags=9
).to(device)


BiLSTM_model.load_state_dict(torch.load('BiLSTM_model.pt'))

# evaluate on the validation set
valid_pred = []

BiLSTM_model.eval()
with torch.no_grad():
  for data, label, len_data, len_label in validloader:
      pred, loss = BiLSTM_model(data.to(device))

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

print("Evaluation on the validation set for task 1:")
precision, recall, f1 = evaluate(
  itertools.chain(*valid_true),
  itertools.chain(*valid_pred))

