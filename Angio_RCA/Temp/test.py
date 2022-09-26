import pandas as pd


df = pd.read_csv('/home/liuwei/Angio/label_processed_anonymized.csv')

columns = ('Patient No.', 'pri_view', 'sec_view', 'npz_filename', 'RCA', 'LMCA', 'LADCA', 'CCA')
df = pd.DataFrame(df, columns=columns)

new_columns = ('PatientID', 'RCA', 'LMCA', 'LADCA', 'CCA',
               'LAOStraight', 'LAOCranial', 'LAOCaudal',
               'RAOStraight', 'RAOCranial', 'RAOCaudal',
               'APStraight', 'APCranial', 'APCaudal',
               'LateralStraight')

new_df = pd.DataFrame(columns=new_columns)

count = 0
for i in range(df.shape[0]):
    # print(i)
    if df.loc[i, 'Patient No.'] not in list(new_df['PatientID']):
        new_df.loc[count] = ''
        new_df.loc[count, 'PatientID'] = df.loc[i, 'Patient No.']
        new_df.loc[count, 'RCA'] = df.loc[i, 'RCA']
        new_df.loc[count, 'LMCA'] = df.loc[i, 'LMCA']
        new_df.loc[count, 'LADCA'] = df.loc[i, 'LADCA']
        new_df.loc[count, 'CCA'] = df.loc[i, 'CCA']
        view = df.loc[i, 'pri_view'] + df.loc[i, 'sec_view']
        new_df.loc[count, view] = df.loc[i, 'npz_filename'] + ', '
        # print(new_df.loc[count])
        # print(count)
        count += 1
    else:
        idx = new_df[new_df['PatientID'] == df.loc[i, 'Patient No.']].index
        print(i)
        view = '{}{}'.format(df.loc[i, 'pri_view'], df.loc[i, 'sec_view'])
        if view not in new_columns:
            print(idx, i, view)
            continue
        new_df.loc[idx, view] = new_df.loc[idx, view] + df.loc[i, 'npz_filename'] + ', '

# print(count)
#
# print(new_df.loc[1])

new_df.to_csv('/home/liuwei/Angio/new_file.csv')
