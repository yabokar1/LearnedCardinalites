Learned Cardinalities in PyTorch
====

PyTorch implementation of multi-set convolutional networks (MSCNs) to estimate the result sizes of SQL queries [1, 2].

## Requirements

  * PyTorch 1.0
  * Python 3.7

## Usage

```python3 train.py --help```

Example usage:

```python3 train.py synthetic```

To reproduce the results in [1] use:

```python3 train.py --queries 100000 --epochs 100 synthetic```

```python3 train.py --queries 100000 --epochs 100 scale```

```python3 train.py --queries 100000 --epochs 100 job-light```

LearnedWMP usage:

1) Execute join_feature_extractor.py to only perform feature extraction for join operators. The script outputs train_tpcds.csv  contains the featurized queries.

```python3 join_feature_extractor.py ```

2) Execute feature_extractor.py to perform feature extraction for join and filter operators (working on including sort, tbscan). The script outputs join_train_tpcds.csv and filter_train_tpcds.csv contains the featurized queries for each operator.

```python3 feature_extractor.py ```

Finally, either from 1) or 2) you need to know # of queries to execute train.py

```python3 train.py --queries 4 --epochs 10 (test set)```


## References

[1] [Kipf et al., Learned Cardinalities: Estimating Correlated Joins with Deep Learning, 2018](https://arxiv.org/abs/1809.00677)

[2] [Kipf et al., Estimating Cardinalities with Deep Sketches, 2019](https://arxiv.org/abs/1904.08223)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2018learned,
  title={Learned cardinalities: Estimating correlated joins with deep learning},
  author={Kipf, Andreas and Kipf, Thomas and Radke, Bernhard and Leis, Viktor and Boncz, Peter and Kemper, Alfons},
  journal={arXiv preprint arXiv:1809.00677},
  year={2018}
}
```
