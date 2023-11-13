import math
import re
import pandas as pd
import sqlparse
from parse import *


QUERIES_MAX_COUNT = 3959
FILENAME_OUTPUT = "train_tpcc.csv"


def feature_extractor(filename):
    df = pd.read_csv(filename)
    df_queries = df[["Queries","db2","actual"]]
    for index,data in df_queries.iterrows():
        # if start >= index:
        #     continue
        if index >= QUERIES_MAX_COUNT:
            break
        actual_mem = data["actual"]
        est_mem = data["db2"]
        if math.isnan(actual_mem) or math.isnan(est_mem):
            continue
        query = data["Queries"]
        query = sqlparse.format(query)
        # print(query)
        query_feature = get_features(query)
        query_feature += "#" + str(est_mem) + "#" + str(actual_mem)
        with open(FILENAME_OUTPUT, "a") as f:
            f.write(query_feature+"\n")
            

def tuple_to_str(tuple):
    tuple_str = ''
    for index,values in enumerate(tuple):
        for value in values:
            if value != '':
                tuple_str += value 
                if index <= (len(tuple) - 1):
                    tuple_str += ","
    return tuple_str 

def list_to_str(list):
    table_list = []
    list_str = ''
    # print(f"list {list}")
    for item in list:
        table_list += item.replace("\n", "").split(",")
    # print(f"table_list is {table_list}")

    for item in table_list:
        words = item.split()
        # print(f"words {words}")
        index = table_list.index(item)
        if len(words) == 2:
            table_list[index] = words[0] + " " + words[1]
            # print(words[0] + "  " + words[1])
        elif 'AS' in words:
            table_list[index] = item.replace("AS", "")
        else:
            table_list[index] = item.replace(" ", "")
        

    list_str = ','.join(table_list)
    list_str  = list_str.rstrip(', ')
    # print(list_str)

    return list_str




def get_features(query):

    table_pattern = r'(?i)FROM\s+([\w, \n]+)(?:\s*(?:AS\s*)?\w*)*(?=\s*WHERE|$)'
    join_pattern = r'\b[A-Za-z]\w*(?:\.[\w\d]+)?\s*=\s*[A-Za-z]\w*(?:\.[\w\d]+)?\b'
    # predicate_pattern = r'\b(\w+)\s*([<>=]=?|=)\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?|\d+))'
    predicate_pattern = r'\b(\w+(?:\.\w+)?)\s*([<>=]=?|=)\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?|\d+))'
    query = sqlparse.format(query)

    table_names = list(set(re.findall(table_pattern, query)))
    join_conditions = set(re.findall(join_pattern, query))
    predicates = set(re.findall(predicate_pattern, query))
   
    print(f"table name is {table_names}")
    # print(f"join is {join_conditions}")
    print(f" predicate is {predicates}")


    predicate_str = tuple_to_str(predicates).replace(" ", "").rstrip(', ')
    join_str = list_to_str(join_conditions)

    table_str = list_to_str(table_names)
   
    query_feature = table_str + "#" + join_str + "#" + predicate_str
    # print(query_feature)


    return query_feature



# sql_query = """
# with wscs as
#  (select sold_date_sk
#         ,sales_price
#   from (select ws_sold_date_sk sold_date_sk
#               ,ws_ext_sales_price sales_price
#         from web_sales 
#         union all
#         select cs_sold_date_sk sold_date_sk
#               ,cs_ext_sales_price sales_price
#         from catalog_sales)),
#  wswscs as 
#  (select d_week_seq,
#         sum(case when (d_day_name='Sunday') then sales_price else null end) sun_sales,
#         sum(case when (d_day_name='Monday') then sales_price else null end) mon_sales,
#         sum(case when (d_day_name='Tuesday') then sales_price else  null end) tue_sales,
#         sum(case when (d_day_name='Wednesday') then sales_price else null end) wed_sales,
#         sum(case when (d_day_name='Thursday') then sales_price else null end) thu_sales,
#         sum(case when (d_day_name='Friday') then sales_price else null end) fri_sales,
#         sum(case when (d_day_name='Saturday') then sales_price else null end) sat_sales
#  from wscs
#      ,date_dim
#  where d_date_sk = sold_date_sk
#  group by d_week_seq)
#  select d_week_seq1
#  where d_week_seq1=d_week_seq2-53
#  order by d_week_seq1;
# """

sql_query = """
with customer_total_return as
(select sr_customer_sk as ctr_customer_sk
,sr_store_sk as ctr_store_sk
,sum(SR_FEE) as ctr_total_return
from store_returns
,date_dim
where sr_returned_date_sk = d_date_sk
and d_year = 2000
group by sr_customer_sk
,sr_store_sk)
 select  c_customer_id
from customer_total_return ctr1
,store
,customer
where ctr1.ctr_total_return > (select avg(ctr_return)*1.2
from customer_total_return ctr2
where ctr1.ctr_store_sk = ctr2.ctr_store_sk)
and s_store_sk = ctr1.ctr_store_sk
and s_state = 'TN'
and ctr1.ctr_customer_sk = c_customer_sk
order by c_customer_id
fetch first 100 rows only;
"""

query = """
 SELECT mi_idx.info AS rating, t.title AS movie_title FROM info_type AS it, keyword AS k, movie_info_idx AS mi_idx, movie_keyword AS mk, title AS t WHERE it.info ='rating' AND k.keyword LIKE 'jane-austen' AND mi_idx.info > '5.0' AND t.production_year > 2005 AND t.id = mi_idx.movie_id AND t.id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND it.id = mi_idx.info_type_id;
"""

# get_features(sql_query)

feature_extractor("../data/tpcc.csv")

# 5397