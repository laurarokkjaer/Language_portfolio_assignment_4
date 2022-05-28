# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#surpress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


    
    
def logistic_regression():
    # Defining filepath 
    filename = os.path.join("..", "..", "CDS-LANG", "toxic", "VideoCommentsThreatCorpus.csv")
    # Reading the data, defining what column i want to start the data with 
    data = pd.read_csv(filename)
    
    # Rename from 1/0 to toxic/non toxic 
    data["label"].replace({0:"non-toxic", 1:"toxic"}, inplace = True)
    
    # Balancing the data for a more even dataset
    data_balanced = clf.balance(data, 1000) # taking 1000 data points
    X = data_balanced["text"]
    y = data_balanced["label"]

    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X,                  # text for the model
                                                        y,                  # classification labels
                                                        test_size = 0.2,    # create an 80/20 split 
                                                        random_state = 42)  # random state for reproducability 

    # Create vectorized object 
    vectorizer = TfidfVectorizer(ngram_range = (1,2),  # 1 = individual words only, 2 = either individual words or bigrams
                             lowercase = True,     # why use lowercase? (so you get all instances of a word)
                             max_df = 0.95,        # df = document frequency. 0.95 = get rid of all words that occur in over 95% of the document
                             min_df = 0.05,        # 0.05 = get rid of all words that occur in lower than 0.5% of the document
                             max_features = 100)   # keep only top 500 features (words)

    #This vectorizer is then used to turn all of our documents into a vector of numbers, instead of text.
    # First we fit to the training data
    X_train_feats = vectorizer.fit_transform(X_train)

    # Then do it for the test data 
    X_test_feats = vectorizer.transform(X_test) # we dont want to fit the test data, because it has to be tested on the training data 

    # Get feature names
    feature_names = vectorizer.get_feature_names()


    # Classifying and predicting 
    # Making my classifier with logistic regression
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats, y_train)

    # Using classifier to predict features 
    y_pred = classifier.predict(X_test_feats)


    # Evaluate
    clf.plot_cm(y_test, y_pred, normalized = True)

    report = metrics.classification_report(y_test, y_pred)
    print(report)

    # Save report 
    with open('../output/logistic_regression.txt', 'w') as my_txt_file:
        my_txt_file.write(report)
    
    print("Script succeeded, results can be seen in output-folder")

logistic_regression()



