# ner-conll2003

This pytorch NLP project explore building deep learning models on named entity recognition (NER) using the CoNLL-2003 dataset.

## Table of Contents

- [Data](#data)
- [Word Embedding](#word-embedding)
- [Models](#models)
- [Evaluation](#evaluation)
- [Instruction](#instruction)

## Data
CoNLL-2003 corpus

## Word Embedding
- Glove Embeddings

```python
# Download Glove Embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```
For the second attempt in train2.py, pretrained “word2vec-google-news-300” Word2Vec model is used.

## Models
- BiLSTM
- Transformer

## Evaluation
evaluate function from [https: //github.com/sighsmile/conlleval](https: //github.com/sighsmile/conlleval)

```python
# Download evalutation script
!wget https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py
```

## Instruction

1. Before running the scripts
To successully run the eval_task1.py and eval_task2.py files, the following code needs to be executed to download required packages.
```
!pip install -q datasets accelerate
!wget -q https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py
!wget -q http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```

Models:
Task 1 model: BiLSTM_model.pt
Task 2 model: BiLSTM_GloVe_model.pt
Bonus model: Transformer_model.pt

2. Command for executing the python files
First of all, to avoid raising error, please put both model files under the same directory as the python files. 

To see evaluation output for task 1, please run the following line.
```
python3 eval_task1.py.py
```

To see evaluation output for task 2, please run the following line.
```
python3 eval_task2.py.py
```

To see evaluation output for bonus task, please run the following line.
```
!python eval_bonus.py
```