import config

import numpy as np
import pandas as pd

import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from string import punctuation
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import text, sequence

import warnings
warnings.simplefilter(action='ignore')

import joblib
from tensorflow.keras.models import load_model

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# Individual pre-processing and training functions
# ================================================
def limit_target_range(df, target):
    return df[target].map(config.TARGET_MAPPINGS)


def date_feature_creation(df, var):
    # convert Date column to datatime object
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Month'] = df[var].apply(lambda x : x.month)
    df['Day'] = df[var].apply(lambda x : x.day)
    df['Weekday'] = df[var].apply(lambda x : x.day_name())
    df['WeekofYear'] = df[var].apply(lambda x : x.weekofyear)  
    ref_date=pd.to_datetime(config.REF_DATE)
    df['Elapsed_Time'] = df[var].apply(lambda x : (x-ref_date).days)
    del df[var]
    return df

####### NON NLP Functions ########
# Rare Labels
def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], 'Others')


# Label Encoder
def label_encode_features(df, var, encoder):
    encoder = joblib.load(encoder)
    df[var]=df[var].apply(lambda x: encoder[x.name].transform(x))
    del encoder
    return df[var]


# Ordinal Encoder
def ordinal_encode_features(df, var, encoder):
    encoder = joblib.load(encoder)
    df[var]=encoder.transform(df[var])
    del encoder
    return df[var]

# MinMax Scaler  
def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    df=pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    del scaler
    return df


####### NLP Functions ########

# List of stop words including punctuation
stop = set(stopwords.words('english'))
punctuation = list(punctuation)
stop.update(punctuation)

#Removing the noisy text
def get_clean_text(text):
    text = text.lower()
    text = re.sub(r"\-", " ", text)
    text=re.sub('\.', ' ', text)
    text = re.sub("\d+", "", text)
    words=nltk.word_tokenize(text)
    words=[word for word in words if not word in set(stop)]
    text = " ".join(words)
    return text


def convert_text2vector(df, var, tokenizer):
    tokenizer=joblib.load(tokenizer)
    df[var]=df[var].apply(get_clean_text)
    df_array = tokenizer.texts_to_sequences(df[var].values)
    df_array = sequence.pad_sequences(df_array, maxlen=config.NLP_MAX_LENGTH)
    del tokenizer
    return pd.DataFrame(df_array, index=df.index)


def predict(data):
    
  data=date_feature_creation(data, config.DATE_VARIABLE)
  
  # Create NLP & Non NLP features
  X_nlp=pd.DataFrame(data[config.TEXT_FEATURE], index=data.index)
  X_nonnlp=data.drop([config.TEXT_FEATURE], axis=1)
  
  ###### Non NLP Data preprocessing ########
  for var in list(config.FREQUENT_LABELS.keys()):
    X_nonnlp[var] = remove_rare_labels(X_nonnlp, var, config.FREQUENT_LABELS[var])
  
  # Label Encoder
  X_nonnlp[config.CATEGORICAL_ENCODE]=label_encode_features(X_nonnlp, config.CATEGORICAL_ENCODE, config.OUTPUT_LABELENCODER_PATH)
  
  # Ordinal Encoder
  X_nonnlp[config.ORDINAL_ENCODE]=ordinal_encode_features(X_nonnlp, config.ORDINAL_ENCODE ,config.OUTPUT_ORDINALENCODER_PATH)
  
  # Scaler
  X_nonnlp=scale_features(X_nonnlp, config.OUTPUT_SCALER_PATH)
  X_nonnlp=np.array(X_nonnlp)
  
  ######### NLP Data preprocessing #########
  X_nlp=convert_text2vector(X_nlp, config.TEXT_FEATURE, config.TOKENIZER_PATH)
  X_nlp=np.array(X_nlp)

  # Predict the data
  # Load saved model
  model=load_model(config.OUTPUT_MODEL_PATH)
  # Predict the results
  prediction=model.predict([X_nlp, X_nonnlp])
  prediction=np.argmax(prediction, axis=1)
  
  del model
  return prediction

    
