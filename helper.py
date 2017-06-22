import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
import settings
import os
import math

def fill_missing_data(frame, column, value = 'None', re_column='None'):
    
    filtered = frame[column].isnull()
    if re_column == 'None':
        frame.loc[filtered, column] = value
    else:
        frame.loc[filtered, column] = frame.loc[filtered, re_column]
    '''
    if re_column == 'None':
        frame[column].fillna(value)
    else:
    	filtered = frame[column].isnull()
        frame.loc[filtered, column] = frame.loc[filtered, re_column]
    '''

def plot_group_prices(frame, column):
    grouped = grouped = frame[[column, 'SalePrice']].groupby(column).mean()
    grouped = grouped.sort_values(by='SalePrice', ascending=0)
    grouped = grouped.reset_index(level=0)

    sns.barplot(grouped[column], grouped["SalePrice"])
    plt.xticks(rotation=30)
    plt.show()
    plt.close()

'''
def encode(frame, column):
    grouped = pd.DataFrame()
    grouped = frame[[column, 'SalePrice']].groupby(column).mean()
    grouped = grouped.sort_values(by='SalePrice')
    grouped['ordering'] = range(1, grouped.shape[0]+1)
    grouped = grouped['ordering'].to_dict()
    
    for cat, o in grouped.items():
        frame.loc[frame[column] == cat, column+'_E'] = o
'''


# Convert categorical features to numeric
def encode(frame):
    str_columns = frame.select_dtypes(include = ["object"]).columns.values.tolist()
    for column in str_columns:
        le = preprocessing.LabelEncoder()
        le.fit(frame[column].unique())
        frame[column] = le.transform(frame[column])

    


def plot_correlation(frame):
    # Pearson correlation 
    corrmat = frame.corr()
    #number of variables for heatmap
    #k = 10 
    #cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

    cols_frame = corrmat['SalePrice'].sort_values(ascending =0)
    cols_frame = cols_frame[(cols_frame>=0.5) | (cols_frame<=-0.5)]
    cols = cols_frame.index

    cm = np.corrcoef(frame[cols].values.T)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
    plt.yticks(rotation=1) 
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

    return cols_frame

def plot_numeric_features(frame, columns):
    #frame = frame[frame.dtypes[(frame.dtypes=="float64")|(frame.dtypes=="int64")].index.values]

    f = pd.melt(frame, value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    plt.show()
    plt.close()


def save_plot_numeric_features(frame, columns, filename):
    f = pd.melt(frame, value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")

    # save the figure to file
    plt.savefig(filename+'.png')   
    plt.close()


def countplot_categorical_features(frame, columns):
    f = pd.melt(frame, value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(countplot, "value")
    g = g.set_ylabels("Count")
    plt.show()
    plt.close()


def save_countplot_categorical_features(frame, columns, filename):
    f = pd.melt(frame, value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=7)
    g = g.map(countplot, "value")
    g = g.set_ylabels("Count")

    # save the figure to file
    plt.savefig(filename+'.png')   
    plt.close()


def countplot(x, **kwargs):
    sns.countplot(x=x)
    plt.xticks(rotation=40)

def boxplot_categorical_features(frame, columns):
    f = pd.melt(frame, id_vars=['SalePrice'], value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(boxplot, "value", "SalePrice")
    plt.show()
    plt.close()

def save_boxplot_categorical_features(frame, columns, filename):
    f = pd.melt(frame, id_vars=['SalePrice'], value_vars=columns)
    g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
    g = g.map(boxplot, "value", "SalePrice")

    # save the figure to file
    plt.savefig(filename+'.png')   
    plt.close()

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    plt.xticks(rotation=45)

def log_transform(frame, columns):
    pd.options.mode.chained_assignment = None 
    frame.loc[:,columns] = np.log1p(frame[columns].values)


def save_csv(frame, filename):
    #move SalePrice to the end
    SalePrice = frame['SalePrice']
    frame = frame.drop(labels=['SalePrice'], axis=1)
    frame.insert(len(frame.columns), 'SalePrice', SalePrice)

    frame.to_csv(os.path.join(settings.PROCESSED_DIR, filename+".csv"), index=False)

def load_csv(filename, location = 'processed'):
    if location == 'processed':
        return pd.read_csv(os.path.join(settings.PROCESSED_DIR, filename+".csv"))
    else:
        return pd.read_csv(os.path.join(settings.DATA_DIR, filename+".csv"))
