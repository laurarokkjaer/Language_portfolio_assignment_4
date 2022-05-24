# simple text processing tools
import re
import os
import sys
sys.path.append(os.path.join(".."))
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

# data wranling
import pandas as pd
import numpy as np

import utils.classifier_utils as clf

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.preprocessing import LabelBinarizer
# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# visualisations 
import matplotlib.pyplot as plt
#%matplotlib inline

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


# Helping functions
def strip_html_tags(text):
  soup = BeautifulSoup(text, "html.parser")
  [s.extract() for s in soup(['iframe', 'script'])]
  stripped_text = soup.get_text()
  stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
  return stripped_text

def remove_accented_chars(text):
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text
def pre_process_corpus(docs):
  norm_docs = []
  for doc in tqdm.tqdm(docs):
    doc = strip_html_tags(doc)
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    # contractions is words like you're vs you are or i'm vs i am or gotta vs got to or yall vs you all 
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    norm_docs.append(doc)
  
  return norm_docs



def deep_learning():
    # Loading dataset
    # defining filepath 
    filename = os.path.join("..", "..", "CDS-LANG", "toxic", "VideoCommentsThreatCorpus.csv")
    # reading the data, defining what column i want to start the data with 
    dataset = pd.read_csv(filename)
    # Rename from 1/0 to toxic/non toxic 
    #dataset["label"].replace({0:"non-toxic", 1:"toxic"}, inplace = True)
    
    # Splitting the data with sklearn, starting of by balancing the data
    data_balanced = clf.balance(dataset, 1000) # taking 1000 data points
    X = data_balanced["text"]
    y = data_balanced["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X,                  # text for the model
                                                    y,                  # classification labels
                                                    test_size = 0.2,    # create an 80/20 split 
                                                    random_state = 42)  # random state for reproducability 
    
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    
    # Defines OOV
    t = Tokenizer(oov_token = '<UNK>' ) # oov = out of vucabulary

    # Fit the tokenizer on the documents
    t.fit_on_texts(X_train_norm)
    # Set padding value, if the text isnt big enough we are doing the following to ensure that the input can fit
    # We pad it and set it to 0, so that the values in the text that we put at the end to make it fit will be 0
    t.word_index['<PAD>'] = 0
    
    # Making the text into a list of integers (numbers)
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    # We want the documents to be the same lengths so they can be used in the classifier 
    # The maximum document length that we will use is going to be a 1000 tokens 
    # This means that the ones that are longer will be cut of and the ones that are shorter will be padded to 1000
    MAX_SEQUENCE_LENGTH = 100

    # Add padding to sequences 
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen = MAX_SEQUENCE_LENGTH)
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen = MAX_SEQUENCE_LENGTH)
    
    
    # Encode lables (what is most efficient in this case, not lb)
    #lb = LabelBinarizer()
    #y_train = lb.fit_transform(y_train)
    #y_test = lb.fit_transform(y_test)
    
    # Clearing the lb model, so that we can use the encoder instead 
    #import tensorflow as tf
    #tf.keras.backend.clear_session()
    
    
    # Define parameters for model 
    # Overall vocabulary size 
    VOCAB_SIZE = len(t.word_index)
    # Number of dimensions for embeddings 
    EMBED_SIZE = 300 
    # Number of epochs to train fro 
    EPOCHS = 2
    # Batch size for training
    BATCH_SIZE = 128

    
    # Create the model
    model = Sequential()
    # Embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    # First convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
    # Print model summary
    #model.summary()
    
    history = model.fit(X_train_pad, y_train,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    validation_split = 0.1,
                    verbose = True)
    
    # Final evaluation of the model 
    scores = model.evaluate(X_test_pad, y_test, verbose = 1)
    #print(f"Accuracy: {scores[1]}")
    
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    #predictions = ["toxic" if item == 1 else "non-toxic" for item in predictions]
    #print(predictions[:10])
    
    # Print classification report 
    labels = ["non-toxic", "toxic"]
    report = classification_report(y_test, predictions, target_names = labels) 
    print(report)
    
    # Save classification report 
    with open('../output/toxic_or_not.txt', 'w') as my_txt_file:
        my_txt_file.write(report)
        
    print("Script succeeded, results can be seen in output-folder")

deep_learning()
        