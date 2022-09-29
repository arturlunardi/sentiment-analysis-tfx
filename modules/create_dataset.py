import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def create_dataset(file_name: str='old_data.xlsx'):
    """
    Save the csv dataset in the data module and create the label encoder file

    Args:
        file_name (str, optional): name of the raw data file. Defaults to 'old_data.xlsx'.

    Returns:
        _type_: None.
    """    
    df = pd.read_excel(f"./data/{file_name}", index_col="Unnamed: 0")
    df = df.loc[(df['sentiment'] != '') & (df['sentiment'].notnull())].reset_index(drop=True)

    df = df[['sentiment', 'title']]
    
    # encode label key to int
    enc = LabelEncoder()
    df['sentiment'] = enc.fit_transform(df['sentiment'])

    # save dataframe to csv
    df.to_csv('./data/data.csv', index=False)

    label_encoder_file = 'label_encoder.pkl'

    # save encoder
    with open(label_encoder_file, 'wb') as output:
        pickle.dump(enc, output)

    return None


if __name__ == '__main__':
    create_dataset()