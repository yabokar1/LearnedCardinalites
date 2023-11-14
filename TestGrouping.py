#%%
import pandas as pd
df = pd.read_csv('data/train_job.csv', header=None)

#%%
df[30] = df[29].str.split('#').apply(pd.Series)[2]
df[30] = df[30].astype(float)
df_clean = df.dropna()
#%%
from sklearn.cluster import KMeans
model = KMeans(n_clusters=15)
classes = model.fit(df_clean[[30]])
classes.labels_
# %%
df_clean['myclass'] = classes.labels_
# %%
df_clean.groupby('myclass').count()
# %%
len(df_clean)
# %%
