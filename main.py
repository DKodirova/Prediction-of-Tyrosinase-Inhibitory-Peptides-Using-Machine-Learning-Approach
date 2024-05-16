from Bio import SeqIO
from libsvm.svmutil import *
import pandas as pd
import numpy as np
import peptides
from sklearn.model_selection import StratifiedKFold

############################### SPLIT DATASETS ############################### 

def loadSeq(filename):
    records = []
    with open(filename, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records.append([record.id, str(record.seq)])
            df = pd.DataFrame(records, columns=['ID', 'Sequence']).set_index('ID')
    return df

def split_by_length(df):
    df1 = df[df['Sequence'].str.len() <= 2]
    df2 = df[df['Sequence'].str.len() >= 3]
    return df1, df2

def save_df_as_fasta(df, filename):
    with open(filename, 'w') as f:
        for k, v in df.agg(''.join, axis=1).items():
            f.write(f'>{k}\n{v}\n')
    return 

train_positive = loadSeq("train_positive.fasta")
train_negative = loadSeq("train_negative.fasta")
test_positive = loadSeq("test_positive.fasta")
test_negative = loadSeq("test_negative.fasta")

train_positive_short, train_positive_long = split_by_length(train_positive)
train_negative_short, train_negative_long = split_by_length(train_negative)
test_positive_short, test_positive_long = split_by_length(test_positive)
test_negative_short, test_negative_long = split_by_length(test_negative)

save_df_as_fasta(train_positive_short, 'data/general_datasets/splitted/train_positive_short.fasta')
save_df_as_fasta(train_negative_short, 'data/general_datasets/splitted/train_negative_short.fasta')
save_df_as_fasta(train_positive_long, 'data/general_datasets/splitted/train_positive_long.fasta')
save_df_as_fasta(train_negative_long, 'data/general_datasets/splitted/train_negative_long.fasta')
save_df_as_fasta(test_positive_short, 'data/general_datasets/splitted/test_positive_short.fasta')
save_df_as_fasta(test_negative_short, 'data/general_datasets/splitted/test_negative_short.fasta')
save_df_as_fasta(test_positive_long, 'data/general_datasets/splitted/test_positive_long.fasta')
save_df_as_fasta(test_negative_long, 'data/general_datasets/splitted/test_negative_long.fasta')


############################### FEATURE SET 1 ############################### 

def load_feature_set1(filename, type):
    records = []
    with open(filename, "r") as handle:
        lines = handle.readlines()
        i = 0
        while i < len(lines):
            header_line = lines[i].strip()
            if header_line.startswith("No."):
                record_id = header_line.split()[1]  
                data_lines = [line.strip() for line in lines[i+1:i+4]] 
                data_values = []
                for line in data_lines:
                    data_values.extend(line.split("\t")) 
                data_values = [float(val) for val in data_values]  
                record_data = data_values[:20] 
                extra_data = data_values[20:]  
                column_names = ['1AA{}'.format(j) for j in range(1, 21)]
                
                if type == "short":
                    extending_columns = ['1F1']
                elif type == "long":
                    extending_columns = ['1F1', '1F2']
        
                column_names.extend(extending_columns)
                records.append([record_id] + record_data + extra_data)
                i += 4  
            else:
                i += 1 
    columns = ["ID"] + column_names
    df = pd.DataFrame(records, columns=columns)
    df.drop(columns=['ID'], inplace = True)
    return df

train_positive_short_feature_set1 = load_feature_set1("data/feature_set1/train_positive_type1_short.txt", "short")
train_positive_long_feature_set1 = load_feature_set1("data/feature_set1/train_positive_type1_long.txt", "long")
train_positive_short_feature_set1["result"] = 1
train_positive_long_feature_set1["result"] = 1

train_negative_short_feature_set1 = load_feature_set1("data/feature_set1/train_negative_type1_short.txt", "short")
train_negative_long_feature_set1 = load_feature_set1("data/feature_set1/train_negative_type1_long.txt", "long")
train_negative_short_feature_set1["result"] = 0
train_negative_long_feature_set1["result"] = 0

train_feature_set1_short = pd.concat((train_positive_short_feature_set1, train_negative_short_feature_set1))
train_feature_set1_long = pd.concat((train_positive_long_feature_set1, train_negative_long_feature_set1))

test_positive_short_feature_set1 = load_feature_set1("data/feature_set1/test_positive_type1_short.txt", "short")
test_positive_long_feature_set1 = load_feature_set1("data/feature_set1/test_positive_type1_long.txt", "long")
test_positive_short_feature_set1["result"] = 1
test_positive_long_feature_set1["result"] = 1

test_negative_short_feature_set1 = load_feature_set1("data/feature_set1/test_negative_type1_short.txt", "short")
test_negative_long_feature_set1 = load_feature_set1("data/feature_set1/test_negative_type1_long.txt", "long")
test_negative_short_feature_set1["result"] = 0
test_negative_long_feature_set1["result"] = 0

test_feature_set1_short = pd.concat((test_positive_short_feature_set1, test_negative_short_feature_set1))
test_feature_set1_long = pd.concat((test_positive_long_feature_set1, test_negative_long_feature_set1))

print("train_positive_short_feature_set1", len(train_positive_short_feature_set1))
print("train_positive_long_feature_set1", len(train_positive_long_feature_set1))
print("train_negative_short_feature_set1", len(train_negative_short_feature_set1))
print("train_negative_long_feature_set1", len(train_negative_long_feature_set1))
print("test_positive_short_feature_set1", len(test_positive_short_feature_set1))
print("test_positive_long_feature_set1", len(test_positive_long_feature_set1))
print("test_negative_short_feature_set1", len(test_negative_short_feature_set1))
print("test_negative_long_feature_set1", len(test_negative_long_feature_set1))

############################### FEATURE SET 2 ############################### 

def load_feature_set2(filename, type):
    rows_num = 4
    if type == "long":
        rows_num = 5
    records = []
    with open(filename, "r") as handle:
        lines = handle.readlines()
        i = 0
        while i < len(lines):
            header_line = lines[i].strip()
            if header_line.startswith("No."):
                record_id = header_line.split()[1]  
                data_lines = [line.strip() for line in lines[i+1:i+rows_num]] 
                data_values = []
                for line in data_lines:
                    data_values.extend(line.split("\t")) 
                data_values = [float(val) for val in data_values]  
                record_data = data_values[:20] 
                extra_data = data_values[20:]  
                column_names = ['2AA{}'.format(j) for j in range(1, 21)]
                column_names.extend(['Hydrophobicity', 'Hydrophilicity', 'Mass', 'pK1 (alpha-COOH)', 'pK2 (NH3)', 'pI (at 25oC)'])
            
                if type == "long":
                    column_names.extend(['Hydrophobicity2', 'Hydrophilicity2', 'Mass2', 'pK1 (alpha-COOH)2', 'pK2 (NH3)2', 'pI (at 25oC)2'])
                
                records.append([record_id] + record_data + extra_data)
                i += rows_num  
            else:
                i += 1 
    columns = ["ID"] + column_names
    df = pd.DataFrame(records, columns=columns)
    df.drop(columns=['ID'], inplace = True)
    return df

train_positive_short_feature_set2 = load_feature_set2("data/feature_set2/train_positive_type2_short.txt", "short")
train_positive_long_feature_set2 = load_feature_set2("data/feature_set2/train_positive_type2_long.txt", "long")
train_positive_short_feature_set2["result"] = 1
train_positive_long_feature_set2["result"] = 1

train_negative_short_feature_set2 = load_feature_set2("data/feature_set2/train_negative_type2_short.txt", "short")
train_negative_long_feature_set2 = load_feature_set2("data/feature_set2/train_negative_type2_long.txt", "long")
train_negative_short_feature_set2["result"] = 0
train_negative_long_feature_set2["result"] = 0

train_feature_set2_short = pd.concat((train_positive_short_feature_set2, train_negative_short_feature_set2))
train_feature_set2_long = pd.concat((train_positive_long_feature_set2, train_negative_long_feature_set2))

test_positive_short_feature_set2 = load_feature_set2("data/feature_set2/test_positive_type2_short.txt", "short")
test_positive_long_feature_set2 = load_feature_set2("data/feature_set2/test_positive_type2_long.txt", "long")
test_positive_short_feature_set2["result"] = 1
test_positive_long_feature_set2["result"] = 1

test_negative_short_feature_set2 = load_feature_set2("data/feature_set2/test_negative_type2_short.txt", "short")
test_negative_long_feature_set2 = load_feature_set2("data/feature_set2/test_negative_type2_long.txt", "long")
test_negative_short_feature_set2["result"] = 0
test_negative_long_feature_set2["result"] = 0

test_feature_set2_short = pd.concat((test_positive_short_feature_set2, test_negative_short_feature_set2))
test_feature_set2_long = pd.concat((test_positive_long_feature_set2, test_negative_long_feature_set2))

############################### FEATURE SET 3 ############################### 

def load_feature_set3(filename):
    records = []
    with open(filename, "r") as handle:
        lines = handle.readlines()
        i = 0
        while i < len(lines):
            header_line = lines[i].strip()
            if header_line.startswith("No."):
                record_id = header_line.split()[1]
                data_lines = [line.strip().split("\t") for line in lines[i+1:i+21]] 
                flat_data = [float(val) for sublist in data_lines for val in sublist]  
                extra_data = flat_data[20:] 
                record_data = flat_data[:20]  
                extra_column_names = ['AACharacter{}'.format(j) for j in range(1, len(extra_data)+1)]
                columns = ["ID"] + ['AA{}'.format(j) for j in range(1, 21)] + extra_column_names
                records.append([record_id] + record_data + extra_data)
                i += 21  
            else:
                i += 1  
                
    df = pd.DataFrame(records, columns=columns)
    df.drop(columns=['ID'], inplace = True)
    return df

train_positive_short_feature_set3 = load_feature_set3("data/feature_set3/train_positive_depeptide_short.txt")
train_positive_long_feature_set3 = load_feature_set3("data/feature_set3/train_positive_depeptide_long.txt")
train_positive_short_feature_set3["result"] = 1
train_positive_long_feature_set3["result"] = 1

train_negative_short_feature_set3 = load_feature_set3("data/feature_set3/train_negative_depeptide_short.txt")
train_negative_long_feature_set3 = load_feature_set3("data/feature_set3/train_negative_depeptide_long.txt")
train_negative_short_feature_set3["result"] = 0
train_negative_long_feature_set3["result"] = 0

train_feature_set3_short = pd.concat((train_positive_short_feature_set3, train_negative_short_feature_set3))
train_feature_set3_long = pd.concat((train_positive_long_feature_set3, train_negative_long_feature_set3))

test_positive_short_feature_set3 = load_feature_set3("data/feature_set3/test_positive_depeptide_short.txt")
test_positive_long_feature_set3 = load_feature_set3("data/feature_set3/test_positive_depeptide_long.txt")
test_positive_short_feature_set3["result"] = 1
test_positive_long_feature_set3["result"] = 1

test_negative_short_feature_set3 = load_feature_set3("data/feature_set3/test_negative_depeptide_short.txt")
test_negative_long_feature_set3 = load_feature_set3("data/feature_set3/test_negative_depeptide_long.txt")
test_negative_short_feature_set3["result"] = 0
test_negative_long_feature_set3["result"] = 0

test_feature_set3_short = pd.concat((test_positive_short_feature_set3, test_negative_short_feature_set3))
test_feature_set3_long = pd.concat((test_positive_long_feature_set3, test_negative_long_feature_set3))


            
############################### ALL FEATURE SETS ############################### 

train_combined_short = pd.concat((train_feature_set1_short, train_feature_set2_short, train_feature_set3_short), axis=1)
train_combined_short = train_combined_short.loc[:,~train_combined_short.columns.duplicated(keep="first")].copy()

train_combined_long = pd.concat((train_feature_set1_long, train_feature_set2_long, train_feature_set3_long), axis=1)
train_combined_long = train_combined_long.loc[:,~train_combined_long.columns.duplicated(keep="first")].copy()

test_combined_short = pd.concat((test_feature_set1_short, test_feature_set2_short, test_feature_set3_short), axis = 1)
test_combined_short = test_combined_short.loc[:,~test_combined_short.columns.duplicated(keep="first")].copy()

test_combined_long = pd.concat((test_feature_set1_long, test_feature_set2_long, test_feature_set3_long), axis = 1)
test_combined_long = test_combined_long.loc[:,~test_combined_long.columns.duplicated(keep="first")].copy()


############################### MODEL ############################### 

def dataframe_to_libsvm(df, target_column, output_file):
    with open(output_file, 'w') as f:
        for index, row in df.iterrows():
            target_value = row[target_column]
            features = [f"{i+1}:{val}" for i, val in enumerate(row.drop(target_column))]
            svm_line = f"{target_value} {' '.join(features)}\n"
            f.write(svm_line)

def apply_model(train, test):
    dataframe_to_libsvm(train, 'result', 'train.txt')
    dataframe_to_libsvm(test, 'result', 'test.txt')
    
    train_y, train_x = svm_read_problem('train.txt')
    test_y, test_x = svm_read_problem('test.txt')

    tips_model = svm_train(train_y, train_x, '-c 4')
    p_label, p_acc, p_val = svm_predict(test_y, test_x, tips_model)
    ACC, MSE, SCC = evaluations(test_y, p_label)
    #print(SCC)
    return ACC, MSE, SCC, p_val

############################### CROSS VALIDATION ############################### 

def apply_cross_validation(df):
    actual_values = []
    predicted_values = []
    for start_row in range(0, len(df), 5):
        end_row = min(start_row + 5, len(df))
        train = pd.concat([df.iloc[:start_row], df.iloc[end_row:]])
        test = df.iloc[start_row:end_row]
        actual_values.extend(test['result'].values)
        
        dataframe_to_libsvm(train, 'result', 'train.txt')
        dataframe_to_libsvm(test, 'result', 'test.txt')
        
        train_y, train_x = svm_read_problem('train.txt')
        test_y, test_x = svm_read_problem('test.txt')

        tips_model = svm_train(train_y, train_x, '-c 4')
        p_label, p_acc, p_val = svm_predict(test_y, test_x, tips_model)
        flat_p_val = [round(item) for sublist in p_val for item in sublist]
        predicted_values.extend(flat_p_val)

    match_count = sum(1 for a, b in zip(actual_values, predicted_values) if a == b)
    similarity = match_count / len(actual_values)
        
    return round(similarity * 100,2)


############################### PERMORMANCE REPORT ############################### 

performance = pd.DataFrame()
performance['Feature Set'] = ['Feature Set #1 (short)', 'Feature Set #1 (long)','Feature Set #2 (short)','Feature Set #2 (long)', 'Feature Set #3 (short)', 'Feature Set #3 (long)', 'All Feature Sets (short)','All Feature Sets (long)']
performance['Type'] = ['Type I (PseACC)', 'Type I (PseACC)', 'Type II (PseACC)', 'Type II (PseACC)', 'Depeptide (PseACC)','Depeptide (PseACC)','N/A', 'N/A']
performance['Lambda'] = ['2','1','2','1','N/A','N/A','N/A','N/A']
performance['Weight'] = ['0.5','0.5','0.5','0.5','N/A','N/A','N/A','N/A']

sets = [[train_feature_set1_short, test_feature_set1_short],
        [train_feature_set1_long, test_feature_set1_long],
        
        [train_feature_set2_short, test_feature_set2_short],
        [train_feature_set2_long, test_feature_set2_long],
        
        [train_feature_set3_short, test_feature_set3_short],
        [train_feature_set3_long, test_feature_set3_long],
        
        [train_combined_short, test_combined_short],
        [train_combined_long, test_combined_long]]

for i in range(len(sets)): 
    performance.at[i, 'Training Size'] = len(sets[i][0])
    performance.at[i, 'Features'] = sets[i][0].columns

    ACC, MSE, SCC, p_val = apply_model(sets[i][0], sets[i][1])
    performance.at[i, 'ACC'] = round(ACC, 2)
    performance.at[i, 'MSE'] = round(MSE, 2)
    performance.at[i, 'SCC'] = round(SCC, 2)
    predictions = pd.DataFrame(p_val, columns = ['prediction'])
    predictions.to_csv(f'predictions/prediction{i}.csv')

    performance.at[i,'Cross Validation Accuracy'] = apply_cross_validation(sets[i][0])
    
performance.to_csv('report.csv')
print(performance)

