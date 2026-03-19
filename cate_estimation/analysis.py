import pandas as pd

# load dataset
data = 'doi_10_5061_dryad_nvx0k6dqg__v20201105/MEH_AMD_survivaloutcomes_database.csv'
df = pd.read_csv(data)

# do EDA of data
print(f'Number of rows: {len(df)}')
print('Columns:\n', df.columns.tolist())
feats_exclude = ['Unnamed: 0', 'X']
print(f'Number of unique patients: {len(df.anon_id.unique())}')
print()
print(f'Gender proportions:\n{df.gender.value_counts()}')
print()
print(f'Ethnicity proportions:\n{df.ethnicity.value_counts()}')
print()
print(f'Age group proportions:\n{df.age_group.value_counts()}')

df = df[(df.date_inj1 == 'Post-2013') & (df.va_inj1 < 70)]
print(len(df), len(df.anon_id.unique()))
print(df.regimen.value_counts())