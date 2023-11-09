import csv
import torch
from torch.utils.data import dataset

from mscn.util import *
import pdb

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
            if int(float(row[3])) < 1:
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


def load_and_encode_train_data(num_queries, num_materialized_samples):
    # file_name_queries = "data/train"
    # file_name_column_min_max_vals = "data/column_min_max_vals.csv"
    file_name_queries = "data/tpcds"
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
            # Convert str into float 
            if type(row[1]) is str and type(row[2]):
                print(row[1])
                print(row[2])
                hash_value_1 = hash(row[1])
                hash_value_2 = hash(row[2])
                row[1] = (hash_value_1 % 1000) 
                row[2] = (hash_value_2 % 1000)
                # print(f"The row 1 {hash_value_1 } and row 2 {hash_value_2}")
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]
    print(f"The column_min_max_val {column_min_max_vals}")
    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)
    
    # return predicates_enc,joins_enc,label_norm,min_val,max_val,samples_enc

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data


def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    print("The sample is {}".format(samples))
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


def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        num_queries, num_materialized_samples)
    
    # print("Exit 1")
    # print(train_data,labels_train,max_num_joins,max_num_predicates)

    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)


    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset
