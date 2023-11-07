import re
import pandas as pd
import sqlparse


QUERIES_MAX_COUNT = 2
FILENAME_OUTPUT = "train_job.csv"


def feature_extractor(filename):
    df = pd.read_csv(filename)
    df_queries = df[["Queries","ESTIMATED_SORT_SHRHEAP_TOP","SORT_SHRHEAP_TOP"]]
    for index,data in df_queries.iterrows():
        if index >= QUERIES_MAX_COUNT:
            break
        actual_mem = data["SORT_SHRHEAP_TOP"]
        est_mem = data["ESTIMATED_SORT_SHRHEAP_TOP"]
        if actual_mem == "nan" or est_mem == "nan":
            continue
        query = data["Queries"]
        query = sqlparse.format(query)
        # print(query)
        query_feature = get_features(query)
        query_feature += "#" + str(est_mem) + "#" + str(actual_mem)
        # with open(FILENAME_OUTPUT, "a") as f:
        #     f.write(query_feature+"\n")
            

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
    print(list)
    for item in list:
        table_list += item.replace("\n", "").split(",")
    print(f"table_list is {table_list}")

    for item in table_list:
        words = item.split()
        print(f"words {words}")
        index = table_list.index(item)
        if len(words) == 2:
            table_list[index] = words[0] + "  " + words[1]
        if 'AS' in words:
            table_list[index] = item.replace("AS", "")
        else:
            table_list[index] = item.replace(" ", "")
        

    list_str = ','.join(table_list)
    list_str  = list_str.rstrip(', ')
    print(list_str)

    return list_str




def get_features(query):
    

    table_pattern = r'(?i)FROM\s+([\w, \n]+)(?:\s*(?:AS\s*)?\w*)*(?=\s*WHERE|$)'
    join_pattern = r'\b([a-zA-Z_]+\s*=\s*[a-zA-Z_]+)\b'
    predicate_pattern = r'\b(\w+)\s*([<>=]=?|=)\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?|\d+))'

    query = sqlparse.format(query, reindent=True, keyword_case='upper')
        # print(query)
    parsed_query = sqlparse.parse(query)[0]
     
    # table_names = set()
    # join_conditions = set()
    # predicates = set()

    for item in parsed_query.tokens:
        print(item)
        if item.match("FROM"):
            print("True")
            
        #     for token in item.tokens:
        #         if token.ttype is None and token.value.strip():
        #             table_names.update(token.value.split(','))
        # elif item.get_real_name() == 'WHERE':
        #     where_clause = str(item).strip()
        #     join_conditions = set(re.findall(join_pattern, where_clause))
        #     predicates = set(re.findall(predicate_pattern, where_clause))

    # table_names = [table.strip() for table in table_names]


    # table_pattern = r'(?i)FROM\s+([\w, \n]+)(?:\s*(?:AS\s*)?\w*)*(?=\s*WHERE|$)'
    # join_pattern = r'\b([a-zA-Z_]+\s*=\s*[a-zA-Z_]+)\b'
    # predicate_pattern = r'\b(\w+)\s*([<>=]=?|=)\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?|\d+))'

    # query = sqlparse.format(query)
    # # print(query)

    # table_names = set(re.findall(table_pattern, query))
    # join_conditions = set(re.findall(join_pattern, query))
    # predicates = set(re.findall(predicate_pattern, query))
   
    # print(table_names)
    # print(join_conditions)
    # print(predicates)
    # print(f"join is {join_conditions}")
    # table_names = list(table_names)

    # predicate_str = tuple_to_str(predicates).replace(" ", "").rstrip(', ')
    # join_str = list_to_str(join_conditions)

    # table_str = list_to_str(table_names)
   
    # query_feature = table_str + "#" + join_str + "#" + predicate_str
    # print(query_feature)


    # return query_feature



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

# sql_query = """
# with customer_total_return as
# (select sr_customer_sk as ctr_customer_sk
# ,sr_store_sk as ctr_store_sk
# ,sum(SR_FEE) as ctr_total_return
# from store_returns
# ,date_dim
# where sr_returned_date_sk = d_date_sk
# and d_year = 2000
# group by sr_customer_sk
# ,sr_store_sk)
#  select  c_customer_id
# from customer_total_return ctr1
# ,store
# ,customer
# where ctr1.ctr_total_return > (select avg(ctr_return)*1.2
# from customer_total_return ctr2
# where ctr1.ctr_store_sk = ctr2.ctr_store_sk)
# and s_store_sk = ctr1.ctr_store_sk
# and s_state = 'TN'
# and ctr1.ctr_customer_sk = c_customer_sk
# order by c_customer_id
# fetch first 100 rows only;
# """

# get_features(sql_query)

feature_extractor("../data/tpcds_master_file.csv")

