import re
import pandas as pd
import sqlparse


QUERIES_MAX_COUNT = 2

def feature_extractor(filename):
    df = pd.read_csv(filename)
    df_queries = df[["Queries","SORT_SHRHEAP_TOP"]]
    for index, data in df_queries.iterrows():
        if index >= QUERIES_MAX_COUNT:
            break
        memory = data["SORT_SHRHEAP_TOP"]
        if memory == "nan":
            continue
        query = data["Queries"]

        query = sqlparse.format(query)
        print(query)
        join_feature, filter_feature = get_features(query)
        join_feature += "#" + str(memory) 
        filter_feature += "#" + str(memory) 
        with open("join_train_tpcds.csv", "a") as f:
            f.write(join_feature+"\n")
        with open("filter_train_tpcds.csv", "a") as f:
            f.write(filter_feature+"\n")
        
            

def tuple_to_str(tuple):
    tuple_str = ''
    for index, values in enumerate(tuple):
        for value in values:
            if value != '':
                tuple_str += value 
                if index <= (len(tuple) - 1):
                    tuple_str += ","
    return tuple_str 

def list_to_str(list):
    list_str = ''
    for index, values in enumerate(list):
        for value in values:
            if value != '':
                list_str += value
        if index < (len(list) - 1):
            list_str += ','
    return list_str




def get_features(query):
    
    table_pattern = r'(?i)FROM\s+([\w, \n]+)(?:\s*(?:AS\s*)?\w*)*(?=\s*WHERE|$)'
    join_pattern = r'\b([a-zA-Z_]+\s*=\s*[a-zA-Z_]+)\b'
    filter_pattern = r'\b(\w+)\s*([<>=]=?|=)\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?|\d+))'

    filter_predicate_str = ''
    join_predicate_str = ''
    sort_predicate_str = ''
    tbscan_predicate_str = ''



    table_names = set(re.findall(table_pattern, query))
    join_predicate = set(re.findall(join_pattern, query))
    filter_predicate = set(re.findall(filter_pattern, query))

    print(f"before {filter_predicate}")

    if filter_predicate:
        filter_predicate_str = tuple_to_str(filter_predicate).replace(" ", "")
    if join_predicate:
        join_predicate_str = list_to_str(join_predicate).replace(" ", "")
    if table_names:
        tablename_str = list_to_str(table_names).replace("\n", "").replace(" ,", ",").strip()



    print(filter_predicate_str)
    print(join_predicate_str)
    print(tablename_str)

    join_feature = tablename_str + "#" + "HS JOIN" + "#" + join_predicate_str
    filter_feature = tablename_str + "#" + "FILTER" + "#" + filter_predicate_str

    return join_feature,filter_feature


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

sql_query = """
 with wscs as
 (select sold_date_sk
        ,sales_price
  from (select ws_sold_date_sk sold_date_sk
              ,ws_ext_sales_price sales_price
        from web_sales 
        union all
        select cs_sold_date_sk sold_date_sk
              ,cs_ext_sales_price sales_price
        from catalog_sales)),
 wswscs as 
 (select d_week_seq,
        sum(case when (d_day_name='Sunday') then sales_price else null end) sun_sales,
        sum(case when (d_day_name='Monday') then sales_price else null end) mon_sales,
        sum(case when (d_day_name='Tuesday') then sales_price else  null end) tue_sales,
        sum(case when (d_day_name='Wednesday') then sales_price else null end) wed_sales,
        sum(case when (d_day_name='Thursday') then sales_price else null end) thu_sales,
        sum(case when (d_day_name='Friday') then sales_price else null end) fri_sales,
        sum(case when (d_day_name='Saturday') then sales_price else null end) sat_sales
 from wscs
     ,date_dim
 where d_date_sk = sold_date_sk
 group by d_week_seq)
 select d_week_seq1
       ,round(sun_sales1/sun_sales2,2)
       ,round(mon_sales1/mon_sales2,2)
       ,round(tue_sales1/tue_sales2,2)
       ,round(wed_sales1/wed_sales2,2)
       ,round(thu_sales1/thu_sales2,2)
       ,round(fri_sales1/fri_sales2,2)
       ,round(sat_sales1/sat_sales2,2)
 from
 (select wswscs.d_week_seq d_week_seq1
        ,sun_sales sun_sales1
        ,mon_sales mon_sales1
        ,tue_sales tue_sales1
        ,wed_sales wed_sales1
        ,thu_sales thu_sales1
        ,fri_sales fri_sales1
        ,sat_sales sat_sales1
  from wswscs,date_dim 
  where date_dim.d_week_seq = wswscs.d_week_seq and
        d_year = 2001) y,
 (select wswscs.d_week_seq d_week_seq2
        ,sun_sales sun_sales2
        ,mon_sales mon_sales2
        ,tue_sales tue_sales2
        ,wed_sales wed_sales2
        ,thu_sales thu_sales2
        ,fri_sales fri_sales2
        ,sat_sales sat_sales2
  from wswscs
      ,date_dim 
  where date_dim.d_week_seq = wswscs.d_week_seq and
        d_year = 2001+1) z
 where d_week_seq1=d_week_seq2-53
 order by d_week_seq1
 """

get_features(sql_query)

# feature_extractor("../data/Master_file.csv")