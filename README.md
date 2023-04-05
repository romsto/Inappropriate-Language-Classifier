# Classification of inappropriate language

## Introduction

The goal of our study is to compare different ML and DL models on classifying text as *appropriate* or *inappropriate language*.
The ultimate goal is to classify in-game chats, but our study will mainly focus on social media texts.

## What is Inappropriate Language ?
We define, for the following work, an *Inappropriate Language* as a word or a sentence that contains any of the following:
- **Profanity** - This includes any language that is considered vulgar, offensive, or obscene. This can include swear words, sexual language, and derogatory terms.
- **Hate speech** - Hate speech is language that is intended to demean, discriminate against, or incite violence or hatred towards a particular group of people based on their race, ethnicity, gender, religion, sexual orientation, or other characteristic.
- **Insults** - This includes any language that is intended to insult or belittle someone else. This can include name-calling, personal attacks, or derogatory comments about someone's appearance, abilities, or personality.
- **Threats** Threats are language that is intended to intimidate or harm another person. This can include physical threats, verbal abuse, or intimidation.



## Models tested

- Decision Tree
    - Source: sklearn
    - Parameters: None
- Random Forest
    - Source: sklearn
    - Parameters: None
- SVC (SVM)
    - Source: sklearn
    - Parameters: None
- LSTM
    - Source: Tensorflow
    - LSTM -> Dense layer (In -> Out (2))
- Transformer (mobile BERT / distil BERT)
    - Source: Huggingface
    - Zero Shot Classification


## Our Dataset

Our dataset is a merge of multiple social media Hate Speech datasets.

### Columns:

- `Text`: text to analyze
- `Class`:
    - `0` - Appropriate
    - `1` - Inappropriate

### Data Sources:
- Hate Speech and Offensive Language (https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv) [tweets]
- Measuring Hate Speech (https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) [social media Reddit Twitter and other]
- https://www.kaggle.com/competitions/detecting-insults-in-social-commentary/
- Jibes and Delights https://github.com/ravsodhi/jibes-and-delights (Insults and Compliments targeting INDIVIDUALS) https://aclanthology.org/2021.woah-1.14.pdf
- https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/ 

### Details:
- English only
- From social media comments
