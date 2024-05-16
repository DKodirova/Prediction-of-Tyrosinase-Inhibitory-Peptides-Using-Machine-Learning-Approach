from Bio import SeqIO
from libsvm.svmutil import *
import pandas as pd
import numpy as np
import peptides
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def loadSeq(filename):
    records = []
    with open(filename, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append([record.id, str(record.seq)])
    return pd.DataFrame(records, columns=['ID', 'Sequence'])

train_positive = loadSeq("data/general_datasets/train_positive.fasta")
train_positive["result"] = 1
train_negative = loadSeq("data/general_datasets/train_negative.fasta")
train_negative["result"] = 0


test_positive = loadSeq("data/general_datasets/test_positive.fasta")
test_positive["result"] = 1
test_negative = loadSeq("data/general_datasets/test_negative.fasta")
test_negative["result"] = 0

df_combined = pd.concat([train_positive, train_negative, test_positive, test_negative])

dfs = [train_positive, train_negative, test_positive, test_negative, df_combined]
i = 0
for df in dfs:
    df['Length'] = df['Sequence'].apply(len)
    plt.figure()
    plt.hist(df['Length'], bins=range(0, 22), edgecolor='black')
    plt.xlabel('Length of Strings')
    plt.ylabel('Frequency')
    plt.title('Length Distribution of Strings')
    plt.grid(True)
    #plt.show()
    plt.savefig(f'graphs/length_distribution/{i}.png')
    plt.close()  
    i += 1