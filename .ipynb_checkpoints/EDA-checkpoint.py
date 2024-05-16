from Bio import SeqIO
from libsvm.svmutil import *
import pandas as pd
import numpy as np


def loadSeq(filename):
    records = []
    with open(filename, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append([record.id, str(record.seq)])
    return pd.DataFrame(records, columns=['ID', 'Sequence'])


positive = loadSeq("positive.fasta")
print(positive.head(5))
print(positive.info())

negative = loadSeq("negative.fasta")
print(negative.head(5))
print(negative.info())



