import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT - filepaths to the messages and categories data sets

    OUTPUT - data frame

    reads in, merges, and outputs a combined dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on = ['id'])
    
    return df


def clean_data(df):
    '''
    INPUT - dataframe

    OUTPUT - cleaned and processed dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    row = categories.loc[0]

    # dropping the last two characters of each string/category name
    category_colnames = [w[:-2] for w in row if ((w.endswith('-0')) | (w.endswith('-1')))]

    categories.columns = category_colnames

    # binary category conversion: set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x:
         1 if x.endswith('-1') else 0)
    
    # replacing the old categories column with categories dataframe
    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df,categories], axis = 1)

    # removing duplicates
    df.drop_duplicates(subset = ['id'], inplace = True)

    return df



def save_data(df, database_filename):
    '''
    INPUT - claned dataframe (df) and database filename

    OUTPUT - True (Success)/ False (Fail)

    Saves the cleaned dataframe into database at the given path
    '''  

    engine = create_engine('sqlite:///{}'.format(database_filename))
    try:
        engine.execute('DROP TABLE IF EXISTS InsertTableName')
        engine.execute('DROP TABLE IF EXISTS Message')
        df.to_sql('Message', engine, index=False)
        return True
    except:
        print('Failed to write dataframe to database')
        return False


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()