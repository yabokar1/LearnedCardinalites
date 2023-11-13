import pandas as pd
import sqlparse
from parse import *

def feature_extractor(filename):
    df = pd.read_csv(filename)
    df_queries = df[["Queries","db2","actual"]]
    for index,data in df_queries.iterrows():
        actual_mem = data["actual"]
        est_mem = data["db2"]
        # if math.isnan(actual_mem) or math.isnan(est_mem):
        #     continue
        query = data["Queries"]
        query = sqlparse.format(query)
        query_list = query.split()
        query_str = ' '.join(query_list)
        print(query_str)
        
        query_str += "#" + str(est_mem) + "#" + str(actual_mem)
        with open("output_tpcds.csv", "a") as f:
            f.write(query_str+"\n")





feature_extractor("../data/tpcds_clean.csv")