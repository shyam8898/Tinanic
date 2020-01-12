import pandas as pd
import numpy as np
def detect_outlier(x):
    q75, q25 = np.nanpercentile(x, [75 ,25])
    iqr = q75 - q25
    up=q75 + (1.5 * iqr)
    lo=q25 - (1.5 * iqr)
    return [i for i in x if i>up or i<lo]

def basic_function(df):
    print("ROWS: {} COLUMNS: {}".format(str(df.shape[0]),str(df.shape[1])))
    print('='*100)
    print(df.info())
    print('='*100)
    print("Descriptive Statistics for numerical columns:")
    print(df.describe())
    print('='*100)
    print("Descriptive Statistics for all columns:")
    print(df.describe(include='object'))
    print('='*100)
    print("Non Null Columns and Counts:")
    null_df=pd.DataFrame(df.isnull().sum())
    null_df.columns=['Count']
    print(null_df[null_df['Count'] > 0])
    print('='*100)
    df_numeric=df.select_dtypes(exclude='object')
    df_num_cols=df_numeric.columns
    for c in df_num_cols:
        print(c)
        print(detect_outlier(df[c]))
        print('-'*100)


if __name__ == "__main__":
    titanic=pd.read_csv('titanic.csv')
    basic_function(titanic)
