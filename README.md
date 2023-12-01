# ner-conll2003

This pytorch NLP project explore building deep learning models on named entity recognition (NER) using the CoNLL-2003 dataset.

## Table of Contents

- [Data](#data)
- [Word Embedding](#word-embedding)
- [Models](#models)
- [Evaluation](#evaluation)

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