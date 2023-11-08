import csv
import torch
from torch.utils.data import dataset
import random

from mscn.util import *


def load_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    with open(file_name + ".csv", newline='') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    # Load bitmaps
    # num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    # with open(file_name + ".bitmaps", 'rb') as f:
    #     for i in range(len(tables)):
    #         four_bytes = f.read(4)
    #         if not four_bytes:
    #             print("Error while reading 'four_bytes'")
    #             exit(1)
    #         num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
    #         bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
    #         for j in range(num_bitmaps_curr_query):
    #             # Read bitmap
    #             bitmap_bytes = f.read(num_bytes_per_bitmap)
    #             if not bitmap_bytes:
    #                 print("Error while reading 'bitmap_bytes'")
    #                 exit(1)
    #             bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
    #         samples.append(bitmaps)
    # print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label

def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            # sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc



def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
   
    return thing2idx, idx2thing


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicates_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

            predicates_enc[i].append(pred_vec)

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    # Updated
    if isinstance(val, str):
        val = random.random()
    else:
        val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def load_and_encode_train_data(num_queries=None, num_materialized_samples=None):
    file_name_queries = "data/tpcds_train"
    file_name_column_min_max_vals = "data/tpcds_column_min_max_vals.csv"

    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

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

    print("The join are {}".format(joins))
    print("The predicates are {}".format(predicates))
    print("The tables are {}".format(tables))
    print("The sample are {}".format(samples))
    print("The label are {}".format(label))

    print("The predicate column are {}".format(column_names))
    print("The predicate encoding is  {}".format(column2vec))
    print("The predicate encoding is  {}".format(idx2column))


    print("The join encoding is  {}".format(join_set))
    print("The join encoding vec is  {}".format(join2vec))
    print("The join encoding vec is  {}".format(idx2join))

    print("The table encoding vec is  {}".format(table2vec))
    print("The join encoding vec is  {}".format(idx2table))





    # Get min and max values for each column
    with open(file_name_column_min_max_vals, newline='') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    return predicates_enc,joins_enc,label_norm,min_val,max_val,samples_enc

    # Split in training and validation samples
    # num_train = int(num_queries * 0.9)
    # num_test = num_queries - num_train

    # samples_train = samples_enc[:num_train]
    # predicates_train = predicates_enc[:num_train]
    # joins_train = joins_enc[:num_train]
    # labels_train = label_norm[:num_train]

    # samples_test = samples_enc[num_train:num_train + num_test]
    # predicates_test = predicates_enc[num_train:num_train + num_test]
    # joins_test = joins_enc[num_train:num_train + num_test]
    # labels_test = label_norm[num_train:num_train + num_test]

    # print("Number of training samples: {}".format(len(labels_train)))
    # print("Number of validation samples: {}".format(len(labels_test)))

    # max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    # max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    # dicts = [table2vec, column2vec, op2vec, join2vec]
    # train_data = [samples_train, predicates_train, joins_train]
    # test_data = [samples_test, predicates_test, joins_test]
    # return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data




load_and_encode_train_data()