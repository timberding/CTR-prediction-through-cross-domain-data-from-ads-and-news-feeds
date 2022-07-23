# -*- coding:utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import datetime
from dateutil.relativedelta import relativedelta

from const import Const

def merge_data(input_dir, ads_df, feeds_df, file_name):
    train_merge_df = pd.merge(ads_df, feeds_df, left_on=['user_id', 'date_flg'],
                              right_on=['u_userId', 'join_date'], how='left')
    train_merge_df['app_score'] = train_merge_df['app_score'].values.astype('float').astype(int)
    train_merge_df.to_csv(path_or_buf=os.path.join(input_dir, file_name), encoding="utf_8_sig", index=False)


def preprocess(input_dir):
    """
    read csv file, merge data from ads and feeds domain

    Args:
        input_dir：input file dir

    Returns:

    """
    print("start preprocess data!")
    train_data_ads_df = pd.read_csv(os.path.join(input_dir, Const.TRAIN_DATA_ADS_CSV_PATH), dtype=str)
    train_data_feeds_df = pd.read_csv(os.path.join(input_dir, Const.TRAIN_DATA_FEEDS_CSV_PATH), dtype=str)
    test_data_df = pd.read_csv(os.path.join(input_dir, Const.TEST_DATA_CSV_PATH), dtype=str)
    train_data_ads_df['date_flg'] = train_data_ads_df.pt_d.str[0:8]
    test_data_df['date_flg'] = test_data_df.pt_d.str[0:8]
    train_data_feeds_df['join_date'] = train_data_feeds_df.e_et.str[0:8]
    train_data_feeds_df['join_date'] = train_data_feeds_df['join_date'].apply(
        lambda x: (datetime.datetime.strptime(x, '%Y%m%d') + relativedelta(days=1)).strftime('%Y%m%d'))

    feeds_feature_name = Const.FEEDS_SPARSE_FEATURE_NAME + Const.FEEDS_SEQUENCE_FEATURE_NAME + ['u_userId', 'join_date']
    train_data_feeds_df = train_data_feeds_df[feeds_feature_name]
    train_data_feeds_df = train_data_feeds_df.groupby(['u_userId', 'join_date'], as_index=False).max()

    print('cross-domain data merge for training data...')
    merge_data(input_dir, train_data_ads_df, train_data_feeds_df, Const.MERGE_TRAIN_DATA_FILE)

    print('cross-domain data merge for testing data...')
    merge_data(input_dir, test_data_df, train_data_feeds_df, Const.MERGE_TEST_DATA_FILE)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input file dir")

    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print("unknown args:%s", unknown)

    input_dir = args.input_dir

    preprocess(input_dir)
