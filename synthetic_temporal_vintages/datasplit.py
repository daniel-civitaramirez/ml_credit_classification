
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



'''
The datasplit function divides the data into one train set and 3 testing sets(t1, t2, t3).
Each test set has a different distribution of the classifier(y), to simulate the effect of 
having different temporal vintages of data.

The printer method shows the results of this distribution.
'''

def datasplit(df, t_size, proportion, label, encoding):
    '''
    df: dataframe
    t_size: test_size data split for training / testing
    label: name of target variable
    enconding: boolean if encoding required
    '''

    if t_size <= 0 or t_size >= 1:
        raise ValueError('invalid test_size for data splitting')
    if proportion <= 0 or proportion > 1:
        raise ValueError('invalid scaling metrics')
    if encoding == True:
        df[label] = LabelEncoder().fit_transform(df[label])

    train, test = train_test_split(df, test_size=t_size, random_state=42)
    #2/3 split for the equaled size training sets
    t1, t = train_test_split(test, test_size=2/3, random_state=42)

    dist = min(t[label].mean(), 1-t[label].mean())
    dic = t[label].value_counts()
    c = -1
    nc = -1
    if dic[0] < dic[1]:
        c = 0
        nc = 1
    else:
        c = 1
        nc = 0
    smaller_dist = dist-(dist*proportion)

    #1/2 split to separate t2 and t3 into equal sizes
    number_of_c_smaller = int((t.shape[0]*0.5*smaller_dist))
    number_of_nc_smaller = int(t.shape[0]*0.5*(1-smaller_dist))

    t_c = t[t[label] == c]
    t_nc = t[t[label] == nc]
    t_c = t_c.sample(n=number_of_c_smaller)
    t_nc = t_nc.sample(n=number_of_nc_smaller)
    t2 = pd.concat([t_c, t_nc])
    t2 = t2.sample(frac=1)
    t = t.drop(t2.index)

    train.reset_index(inplace=True)
    t1.reset_index(inplace=True)
    t2.reset_index(inplace=True)
    t.reset_index(inplace=True)

    return train, t1, t2, t


def printer(df, train, t1, t2, t3, label):
    print('Full Dataset: ', df.shape)
    print('Training shape: ', train.shape)
    print('t1 shape: ', t1.shape)
    print('t2 shape: ', t2.shape)
    print('t3 shape: ', t3.shape)
    print('% of Dataset: ', [train.shape[0]/df.shape[0], t1.shape[0] /
                             df.shape[0], t2.shape[0]/df.shape[0], t3.shape[0]/df.shape[0]])
    print()
    print('Training Classifier Distribution:', min(
        train[label].mean(), 1-train[label].mean()))
    print(train[label].value_counts())
    print()
    print('t1 Classifier Distribution:', min(
        t1[label].mean(), 1-t1[label].mean()))
    print(t1[label].value_counts())
    print()
    print('t2 Classifier Distribution:', min(
        t2[label].mean(), 1-t2[label].mean()))
    print(t2[label].value_counts())
    print()
    print('t3 Classifier Distribution:', min(
        t3[label].mean(), 1-t3[label].mean()))
    print(t3[label].value_counts())
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title('Distribution of Classifier in Tables')
    ax.set_xlabel('Table')
    ax.set_ylabel('Fraction of Classifer')
    l = ['train', 't1', 't2', 't3']
    v = [min(train[label].mean(), 1-train[label].mean()), min(t1[label].mean(), 1-t1[label].mean()),
         min(t2[label].mean(), 1-t2[label].mean()), min(t3[label].mean(), 1-t3[label].mean())]
    ax.bar(l, v)
    plt.show()
