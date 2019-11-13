import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sklearn
import nltk

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import multiclass
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


import re

nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer



def load_data(database_filepath):
    '''
    INPUT - database file path

    OUTPUT - X, Y as data and labels

    Reads in the dataframe from the database and outputs the data (X) and labels (Y)
    '''

    # setting up the DB engine and reading in the data
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table(table_name = 'Message', con = conn)

    # extracting X and Y, flattening X into a 1D array
    X = df[['message']]
    Y = df.iloc[:,5:41]
    X.reset_index(drop = True, inplace = True)
    X = X.values.flatten()

    return X, Y


def tokenize(text):
    '''
    INPUT - the text of a message

    OUPUT - list of tokens after normalization and lemmatization
    '''
    text = re.sub("[^A-Za-z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(t).strip() for t in tokens]
    return lemmed

class MessageTypeExtractor(BaseEstimator, TransformerMixin):
    '''
    Estimator/Transformer class for identifying text that contains numerical values
    and transforming them into binary features
    '''

    def contains_num(self, text):
        '''
        INPUT - text of the message

        OUPUT - 1 if message contains numerical values or else 0
        '''
        if len(re.findall('[0-9]', text)) != 0:
            return 1
        else:
            return 0
    
    def fit(self, s, y=None):
        '''
        Signature for the required fit(*args) method inherited from BaseEstimator
        '''
        return self
    
    def transform(self, X):
        '''
        INPUT - Unlabled data (X) 

        OUPUT - Transformed data in form of binary features based on contains_num()
        '''
        Xnum = pd.Series(X).apply(self.contains_num)
        return Xnum


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()