import config

import numpy as np
import pandas as pd

import re

import nltk
nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import text, sequence
from imblearn.over_sampling import SMOTE

import warnings
warnings.simplefilter(action='ignore')

import joblib
from tensorflow.keras.models import load_model

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, concatenate, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

nltk.download('punkt')


# Individual pre-processing and training functions
# ================================================


def load_data(df_path):
    # Function loads data for training
    df=pd.read_csv(df_path, index_col=0)
    df.rename(columns={'Data': 'Date', 'Countries':'Country', 'Genre':'Gender', 'Employee or Third Party':'Employee type'}, inplace=True)
    del df['Accident Level']
    return df


def limit_target_range(df, target):
    return df[target].map(config.TARGET_MAPPINGS)


def date_feature_creation(df, var):
    # convert Date column to datatime object
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df[var].apply(lambda x : x.month)
    df['Day'] = df[var].apply(lambda x : x.day)
    df['Weekday'] = df[var].apply(lambda x : x.day_name())
    df['WeekofYear'] = df[var].apply(lambda x : x.weekofyear)  
    ref_date=pd.to_datetime(config.REF_DATE)
    df['Elapsed_Time'] = df[var].apply(lambda x : (x-ref_date).days)
    del df[var]
    return df

    
#Split dataset
def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), df[target], test_size=config.TEST_SIZE, random_state=config.SEED)
    return X_train, X_test, y_train, y_test


####### NON NLP Functions ########
# Rare Labels
def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], 'Others')


# Label Encoder
def train_label_encoder(df, var ,output_path):
    encoder = defaultdict(LabelEncoder)
    df[var].apply(lambda x: encoder[x.name].fit(x))
    joblib.dump(encoder, output_path)
    return encoder

def label_encode_features(df, var, encoder):
    encoder = joblib.load(encoder)
    df[var]=df[var].apply(lambda x: encoder[x.name].transform(x))
    return df[var]


# Ordinal Encoder
def train_ordinal_encoder(df, var ,output_path):
    encoder = OrdinalEncoder()
    encoder.fit(df[var])
    joblib.dump(encoder, output_path)
    return encoder

def ordinal_encode_features(df, var, encoder):
    encoder = joblib.load(encoder)
    df[var]=encoder.transform(df[var])
    return df[var]

# MinMax Scaler
def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
  
def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    df=pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    return df

#Drop unwanted features
def drop_unnescessary_features(df, var):
    return df.drop(var, axis=1)


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

def train_tokenizer(df, var, output_path):
    df[var]=df[var].apply(get_clean_text)
    tokenizer = text.Tokenizer(num_words=config.NLP_VOCAB_SIZE)
    tokenizer.fit_on_texts(df[var])
    joblib.dump(tokenizer, output_path)
    return tokenizer

def convert_text2vector(df, var, tokenizer):
    tokenizer=joblib.load(tokenizer)
    df[var]=df[var].apply(get_clean_text)
    df_array = tokenizer.texts_to_sequences(df[var].values)
    df_array = sequence.pad_sequences(df_array, maxlen=config.NLP_MAX_LENGTH)
    return pd.DataFrame(df_array, index=df.index)

def upsample(df_nlp, df_nonnlp, df_target):
    df=np.concatenate( (np.array(df_nlp), np.array(df_nonnlp)), axis=1 )
    upsamples_count=config.UPSAMPLE_COUNT
    
    oversample = SMOTE(random_state=config.SEED, sampling_strategy={0:upsamples_count, 1:upsamples_count, 2:upsamples_count, 3:upsamples_count, 4:upsamples_count})
    df, df_target = oversample.fit_resample(df, df_target)
    
    df_nlp=df[:,:config.NLP_MAX_LENGTH]
    df_nonnlp=df[:,config.NLP_MAX_LENGTH:]
    return df_nlp, df_nonnlp, df_target


def create_embedding_matrix(wv_model, tokenizer, output_path):
    tokenizer=joblib.load(tokenizer)
    wv_model=joblib.load(wv_model)
    
    embedding_matrix = np.zeros((config.NLP_VOCAB_SIZE, config.NLP_EMBEDDING_SIZE))
    for word, i in list(tokenizer.word_index.items())[:config.NLP_VOCAB_SIZE-1]:
        if word in wv_model.wv.vocab:
            embedding_vector = wv_model.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        else:
          print(word)
    
    joblib.dump(embedding_matrix, output_path)
    return embedding_matrix

# Train Model
def train_model(df_nlp, df_nonnlp, df_target, df_test_nlp, df_test_nonnlp, df_test_target ,embedding_matrix ,output_path): 
    
    # model inputs
    nlp_input=Input(shape=df_nlp.shape[1])
    nonnlp_input=Input(shape=df_nonnlp.shape[1])
    
    #load embeddings
    embedding=Embedding(config.NLP_VOCAB_SIZE, output_dim=config.NLP_EMBEDDING_SIZE, 
                        trainable=False, weights=[embedding_matrix])(nlp_input)
    
    # LSTM Layers
    nlp_out=Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.2, activation='tanh', return_sequences=True))(embedding)
    nlp_out=LSTM(512, dropout=0.2, recurrent_dropout=0.2, activation='tanh')(nlp_out)
    
    # Concat NLP + NONNLP inputs
    nlp_nonnlp_out=concatenate([nlp_out, nonnlp_input])
    
    # Dense layers
    nlp_nonnlp_out=Dense(512, activation='relu')(nlp_nonnlp_out)
    nlp_nonnlp_out=Dropout(.2)(nlp_nonnlp_out)
    nlp_nonnlp_out=Dense(256, activation='relu')(nlp_nonnlp_out)
    nlp_nonnlp_out=Dropout(.2)(nlp_nonnlp_out) 
    nlp_nonnlp_out=Flatten()(nlp_nonnlp_out)
    
    # Output classes
    output=Dense(5,activation='softmax')(nlp_nonnlp_out)
    
    # Deep Learning Model
    model=Model(inputs=[nlp_input, nonnlp_input], outputs=[output])
    # Model Compile
    model.compile(optimizer=Adam(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Model Train
    checkpoint = ModelCheckpoint(config.OUTPUT_MODEL_PATH, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=False)
    stop = EarlyStopping(monitor="val_loss", patience=7)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    model.fit([df_nlp, df_nonnlp], df_target, epochs = config.MODEL_EPOCHS, batch_size=32, verbose = 2, 
              validation_data=([df_test_nlp, df_test_nonnlp], df_test_target),
              callbacks=[checkpoint, stop, reduce_lr])
    
    # save the model - best model is saved during checkpoint
    #model.save(config.OUTPUT_MODEL_PATH)
    
    return model


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
  
  return prediction

    
