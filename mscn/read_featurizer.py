import csv
import pandas as pd

file_name = "../data/train_job"

df = pd.DataFrame(columns=['tablename', 'join', 'predicate', 'estimate', 'actual'])
with open(file_name + ".csv", newline='') as f:
    data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
    for row in data_raw:
        df.loc[len(df)] = row






# for index, row in df.iterrows():
    # print(row['tablename'])
    # print(row['join'])
    # print(row['predicate'])
    # print(row['estimate'])
    # print(row['actual'])

# actual_est = df[["actual"]]

# Update 'actual' column
df['actual'] = df['actual'].astype(float) * 4000 / 1000000

# Update 'estimate' column
df['estimate'] = df['estimate'].astype(float) * 4000 / 1000000

# Print the updated DataFrame
# print(df)


df.to_csv("updated_train_job.csv")