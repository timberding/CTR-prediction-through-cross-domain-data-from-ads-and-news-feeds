# -*- coding:utf-8 -*-


class Const:
    """
        define const
    """
    TRAIN_DATA_ADS_CSV_PATH = 'train/train_data_ads.csv'
    TRAIN_DATA_FEEDS_CSV_PATH = 'train/train_data_feeds.csv'
    TEST_DATA_CSV_PATH = 'test/test_data_ads.csv'
    MERGE_TRAIN_DATA_FILE = 'train/train_data_file.csv'
    MERGE_TEST_DATA_FILE = 'test/test_data_file.csv'
    PREDICT_OUT_FILE = 'submission.csv'
    CHECK_POINT_PATH = 'checkpoint'
    CONFIG_FILE = 'config.json'

    EMBEDDING_SIZE = 'embedding_size'
    BATCH_SIZE = 'batch_size'
    EPOCHS = 'epochs'
    VALIDATION_SPLIT = 'validation_split'
    VERBOSE = 'verbose'
    SHUFFLE = "isShuffle"
    SEQ_MAX_LEN = 'seq_max_len'
    HASH_FLG = "hashFlg"
    POOLING_MODE = 'pooling'
    CROSS_NUM = 'cross_num'
    CROSS_PARAMETERIZATION = 'cross_parameterization'
    HIDDEN_SIZE = 'hidden_size'
    EARLY_STOP_PATIENCE = 'early_stop_patience'
    EARLY_STOP_DEPENDENCY = 'early_stop_dependency'
    DNN_DROPOUT = "dnn_dropout"
    OPTIMIZER = 'optimizer'
    LOSS = 'loss'

    AUC_THRESHOLDS = 'auc_thresholds'
    METRICS = 'metrics'

    ADS_SPARSE_FEATURE_NAME = ['age', 'gender', 'residence', 'city', 'city_rank', 'series_dev', 'series_group',
                               'emui_dev',
                               'device_name', 'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd',
                               'adv_prim_id',
                               'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id', 'hispace_app_tags',
                               'app_second_class', 'app_score']
    ADS_SEQUENCE_FEATURE_NAME = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003', 'ad_close_list_v001',
                                 'ad_close_list_v002', 'ad_close_list_v003']

    FEEDS_SPARSE_FEATURE_NAME = ['u_phonePrice', 'u_browserLifeCycle', 'u_browserMode', 'u_feedLifeCycle',
                                 'u_refreshTimes']

    FEEDS_SEQUENCE_FEATURE_NAME = ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news']
