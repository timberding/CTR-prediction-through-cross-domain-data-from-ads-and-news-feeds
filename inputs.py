# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

from collections import defaultdict
from itertools import chain

from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from .layers.utils import Hash


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    """
    create embedding of sparse and sequence features
    Args:
        sparse_feature_columns: sparse feature list, define type of feature
        varlen_sparse_feature_columns: sequence feature list, define type of feature
        seed: seed
        l2_reg：
        prefix: prefix
        seq_mask_zero：for sequence features, whether to mask zero

    Returns:
        sparse_emb_dict: raw embedding dict

    """
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    """
    create embedding of sparse and sequence features
    Args:
        feature_columns: feature list, define type of feature
        l2_reg: l2_reg
        seed: seed
        prefix: prefix
        seq_mask_zero：for sequence features, whether to mask zero

    Returns:
        sparse_emb_dict: embedding dict

    """

    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns,
                     mask_feat_list=(), to_list=False):
    """
    get embedding of feature
    Args:
        sparse_embedding_dict: embedding dict
        sparse_input_dict: input layer
        sparse_feature_columns: feature list, define type of feature
        mask_feat_list: feature list for mask
        to_list: convert to list flag
    Returns:
        group_embedding_dict: group embedding dict

    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size,
                              mask_zero=(feature_name in mask_feat_list),
                              vocabulary_path=fc.vocabulary_path)(
                sparse_input_dict[feature_name])
        else:
            lookup_idx = sparse_input_dict[feature_name]
        group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    """
    get embedding of sequence feature
    Args:
        embedding_dict: embedding dict
        sequence_input_dict: input layer
        varlen_sparse_feature_columns: feature list, define type of feature

    Returns:
        varlen_embedding_vec_dict: group embedding dict

    """
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size,
                              mask_zero=True,
                              vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    """
    get embedding after pooling for sequence feature
    Args:
        embedding_dict: embedding dict
        features: input layer
        varlen_sparse_feature_columns: feature list, define type of feature

    Returns:
        pooling_vec_list: group embedding dict after pooling

    """
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    from . import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def mergeDict(a, b):
    """
    merge layer
    Args:
        a: input a
        b: input b

    Returns:
        c: output after merge

    """
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
