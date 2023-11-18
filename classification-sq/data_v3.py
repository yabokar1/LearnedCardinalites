import csv
import pandas as pd
import torch
from torch.utils.data import dataset

from util_v3 import *


def load_data(file_name):
    joins = []
    predicates = []
    tables = []
    samples = []
    labels = []

    df = pd.read_csv(file_name, delimiter='#')

    df = df[df['actual'] >0]

    # print(df.dtypes)

    num_queries = df.shape[0]

    # how to iterate rows in a dataframe?
    # Generated by WCA for GP
    for index, row in df.iterrows():
        tables.append(row['tables'].split(','))
        joins.append(row['joins'].split(','))
        predicates.append(str(row['predicates']).split(','))
        labels.append(row['template'])

    print("Loaded queries")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]


    return joins, predicates, tables, samples, labels, num_queries


def load_and_encode_train_data(dataset_name):

    # SQ: renamed the train.csv (which had the training dataset for job) to train_job.csv
    # SQ: added tpcds train dataset to the data folder and changing the file name below for tpcds
    file_name_queries = "train_{}.csv".format(dataset_name)

    # SQ: changed the following code to read the columns min and max for the tpcds dataset
    file_name_column_min_max_vals = "{}_column_min_max_vals.csv".format(dataset_name)

    joins, predicates, tables, samples, labels, num_queries = load_data(file_name_queries)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    # SQ: changed file open model from rU to r+
    with open(file_name_column_min_max_vals, 'r+') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            # SQ: the following code is checking the data types of the column's min and max values
            # If these values are categorical (str), we're hashing them to generate a numeric value
            if type(row[1]) is str and type(row[2]) is str:
                hash_value_1 = hash(row[1])
                hash_value_2 = hash(row[2])
                row[1] = (hash_value_1 % 1000)
                row[2] = (hash_value_2 % 1000)
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    # label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = labels[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = labels[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    return dicts, column_min_max_vals, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data


def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    # print("The sample is {}".format(samples))
    for sample in samples:
        sample_tensor = np.vstack(sample)
        # num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        num_pad = max_num_joins + 6 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        # print("The sample tensor is {}, {}".format(max_num_joins,sample_tensor.shape[0]))
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
                                 predicate_masks, join_masks)


def get_train_datasets(dataset_name):
    dicts, column_min_max_vals, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = load_and_encode_train_data(dataset_name)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins, max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset