import pandas as pd
import cv2
import os


# df = pd.read_csv('/home/liuwei/Angio/new_file.csv')
#
# print(df.shape[0])
# print(type(df.loc[300, 'RCA']))
#
# idx = []
# for i in range(df.shape[0]):
#     if isinstance(df.loc[i, 'RCA'], float) and isinstance(df.loc[i, 'LMCA'], float) \
#             and isinstance(df.loc[i, 'LADCA'], float) and isinstance(df.loc[i, 'CCA'], float):
#         print(df.loc[i, ['RCA', 'LMCA', 'LADCA', 'CCA']])
#         idx.append(i)
#
# df = df.drop(idx)
#
# print(len(idx), df.shape[0])
#
# df.to_csv('/home/liuwei/Angio/new_file1.csv')
###########################################################

# df = pd.read_csv('/home/liuwei/Angio/new_file1.csv')
# et_df = pd.read_csv('/home/liuwei/Angio/20210611_LAO_Straight.csv')
#
# print(df.shape[0])
#
# pid = []
# for i in range(et_df.shape[0]):
#     # if isinstance(et_df.loc[i, 'Lines'], float) and isinstance(et_df.loc[i, 'Artifacts'], float):
#     if et_df.loc[i, 'Lines'] == 1 and et_df.loc[i, 'Artifacts'] == 1:
#         # print(et_df.loc[i])
#         pid.append(et_df.loc[i, 'Patient No.'])
#
# print(len(pid))
#
# print(df.columns)
# idx = []
# for i in range(df.shape[0]):
#     if df.loc[i, 'PatientID'] not in list(et_df['Patient No.']):
#         idx.append(i)
#         # print(df.loc[i, 'PatientID'])
#     else:
#         if df.loc[i, 'PatientID'] in pid:
#             # print(df.loc[i, 'PatientID'])
#             idx.append(i)
#
#
# df = df.drop(idx)
# print(pid)
# print(len(idx), df.shape[0])
# df.to_csv('/home/liuwei/Angio/new_file2.csv')
#################################################################

# ip_npz_path = '/home/liuwei/local_mnt/Angio/data_26-8-2020/NHC Processed Data/'
# op_npz_path = ''
#
# if not os.path.exists(op_path):
#     os.makedirs(op_path)