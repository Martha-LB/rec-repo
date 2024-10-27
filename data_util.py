# Artur Andrzejak, October 2024
# Data utilities for recommender systems

import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


# Routines to load MovieLens data from the original site and convert it to utility matrix (polars, numpy)

def get_size_in_mb(obj):
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size(unit='mb')
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    else:
        return None


def print_df_stats(prefix_msg, df):
    print(f"{prefix_msg}, df shape: {df.shape}, size in MB: {get_size_in_mb(df)} ")


def read_movielens_file_and_convert_to_um(file_path, max_rows=None):
    """ Read a local MovieLens file and return the utility matrix as a pivot table where
        columns=userid, rows=movieid (relabeled) and contents are movie rating (NaN = no rating given).
        Original file has columns: userId,movieId,rating,timestamp
        See https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """
    print(f"\n### Start reading data from '{file_path}'")
    df = pl.read_csv(file_path,
                     has_header=True, columns=[0, 1, 2],
                     new_columns=['userID', 'movieID', 'rating'],
                     n_rows=max_rows,
                     schema_overrides={'userID': pl.UInt32, 'movieID': pl.UInt32, 'rating': pl.Float32()})
    print_df_stats(f"Loaded data from '{file_path}'", df)

    # Convert from long to wide format and then drop column 'movieID'
    print("Pivoting the data")
    util_mat_pl = df.pivot(index='movieID', on='userID', values='rating').drop('movieID')
    print_df_stats(f"Utility matrix", util_mat_pl)

    util_mat_np = np.array(util_mat_pl).astype(np.float32)
    print_df_stats(f"Final utility matrix (numpy array as np.float32)", util_mat_np)
    return util_mat_np


def load_and_unzip_dataset(url, path_to_save, unzip_path, force_download=False):
    import requests
    import zipfile
    import io, os

    if not force_download and os.path.exists(unzip_path):
        print(f"Dir '{unzip_path}' already exists, skipping download")
        return

    print(f"Downloading '{url}' to '{path_to_save}'")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_save)
    print(f"Loaded and unzipped '{url}' to '{unzip_path}'")


def get_um_by_name(config, dataset_name):
    if dataset_name == "movielens":
        # Load (part of) the MovieLens 25M data
        # See https://grouplens.org/datasets/movielens/25m/
        load_and_unzip_dataset(config.dowload_url, config.download_dir, config.unzipped_dir)
        return read_movielens_file_and_convert_to_um(
            config.file_path,
            max_rows=config.max_rows)
    elif dataset_name == "lecture_1":
        um_lecture = [[    1., np.nan,     3., np.nan, np.nan,     5.],
                      [np.nan, np.nan,     5.,     4., np.nan, np.nan],
                      [    2.,     4., np.nan,     1.,     2., np.nan],
                      [np.nan,     2.,     4., np.nan,     5., np.nan],
                      [np.nan, np.nan,     4.,     3.,     4.,     2.],
                      [    1., np.nan,     3., np.nan,     3., np.nan]]
        return np.asarray(um_lecture)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# Routines to load and preprocess MovieLens dataset from TensorFlow Datasets

def print_sample_of_tf_dataset(ds, message=None, num_samples=5):
    if message:
        print(message)
    for i, elem in enumerate(ds.take(num_samples)):
        print(f"Sample {i}: {elem}")


def load_movielens_tf(config):
    """ Load the MovieLens dataset using TensorFlow Datasets and preprocess it
        See https://www.tensorflow.org/ranking/tutorials/quickstart for alternative processing
    """

    postfix = '-ratings'
    ratings, info = tfds.load(config.dataset_base_name + postfix,
                              split=config.dataset_split,
                              shuffle_files=config.shuffle_files,
                              data_dir=config.data_dir,
                              with_info=True)
    print(
        f"Loaded dataset '{config.dataset_base_name}' with {ratings.cardinality()} ratings and features: {info.features}")
    assert isinstance(ratings, tf.data.Dataset) #确保 ratings 是一个 tf.data.Dataset 对象

    # The dataset has more features, but we only need the following
    print("Filtering tf dataset for user_id, movie_id and user_rating")
    # map函数遍历数据集的每一条记录，将指定的函数应用于每个元素，并返回一个新的数据集
    # lambda 是一种用于定义匿名函数的方式，x 代表数据集中每一条记录，是一个字典
    # 代码块中的大括号 {} 表示一个字典
    # 在这里map遍历ratings数据集中每一条数据，每条数据都被传递给匿名函数lambda，赋值给x，然后根据x提取中其中的user_id等等，构建一个新字典
    ratings = ratings.map(lambda x: {
        "user_id": x["user_id"],
        "movie_id": x["movie_id"],
        "user_rating": x["user_rating"],
    })

    print("Creating a vocabulary for user_id (str -> int)")
    # user_id_data 的类型是 tf.data.Dataset
    # StringLookup 是 TensorFlow Keras 提供的一种层，主要用于将字符串映射为整数索引
    user_id_data = ratings.map(lambda x: x["user_id"])
    user_ids_voc = tf.keras.layers.StringLookup(mask_token=None)
    #在实例化 StringLookup 后，你需要调用其 adapt 方法，该方法会根据给定的数据（如 user_id_data）创建一个字典，将每个唯一的字符串映射为一个唯一的整数索引
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup#adapt
    user_ids_voc.adapt(user_id_data.batch(10000)) #  处理前 10000 个用户 ID，并构建相应的映射
    # get_vocabular方法返回 StringLookup 层内部维护的词汇表，包含所有唯一的用户 ID 及其对应的整数索引
    print(f"Vocabulary of user_id's has size: {len(user_ids_voc.get_vocabulary())}")

    print("Creating a vocabulary for movie_id (str -> int)")
    movie_id_data = ratings.map(lambda x: x["movie_id"])
    movie_ids_voc = tf.keras.layers.StringLookup(mask_token=None)
    movie_ids_voc.adapt(movie_id_data.batch(10000))
    print(f"Vocabulary of movie_id's has size: {len(movie_ids_voc.get_vocabulary())}")

    # Map the string user_id and movie_id to integer indices
    # user_ids_voc(x["user_id"]) 将用户 ID 传入 StringLookup 层，返回对应的整数索引
    ratings = ratings.map(lambda x: {
        "user_id": user_ids_voc(x["user_id"]),
        "movie_id": movie_ids_voc(x["movie_id"]),
        "user_rating": x["user_rating"]
    })
    # 此时，ratings是一个dataset，每条记录都是一条字典，只是原本的值由字符被转化成了整数
    # user_ids_voc, movie_ids_voc的类型是tf.keras.layers.StringLookup，分别是用于用户 ID 字符串到整数索引映射的对象和用于电影 ID 字符串到整数索引映射的对象，通过调用 get_vocabulary() 方法，可以获取映射的详细信息
    # 举例：movie_ids_voc("movie_1") 会返回 1
    return ratings, user_ids_voc, movie_ids_voc


def split_train_valid_test_tf(ratings_tf, split_ratios):
    """ Split the dataset into train, validation and test sets
    Args:
        ratings_tf: the dataset
        split_ratios: a tuple with the fractions of the dataset to use for train, validation and test
    Returns:
        train_ds, valid_ds, test_ds: the datasets
    """
    # cardinality() 方法用于获取数据集中元素的数量。它返回一个 tf.int64 类型的张量, numpy()将张量转化为一个numpy数组，便于查看
    num_samples = ratings_tf.cardinality().numpy()
    print(f"Splitting the dataset into train, validation and test sets with sizes: {split_ratios}")

    # Shuffle the dataset
    # shuffle方法会对数据集的元素进行随机排列（打乱），设置seed保证数据集在每次运行时被打乱的顺序相同，False表示数据集在每次训练迭代时保持相同的打乱顺序
    ratings_tf = ratings_tf.shuffle(num_samples, seed=42, reshuffle_each_iteration=False)

    # Split the dataset
    train_size = int(num_samples * split_ratios[0])
    valid_size = int(num_samples * split_ratios[1])
    test_size = num_samples - train_size - valid_size

    train_ds = ratings_tf.take(train_size) # 从数据集 ratings_tf 中提取前 train_size 个样本
    valid_ds = ratings_tf.skip(train_size).take(valid_size) #跳过 train_size 个样本，从第 train_size+1 个样本开始
    test_ds = ratings_tf.skip(train_size + valid_size).take(test_size)
    print(
        f"Absolute sizes => Train: {train_ds.cardinality()}, Validation: {valid_ds.cardinality()}, Test: {test_ds.cardinality()}")
    return train_ds, valid_ds, test_ds


if __name__ == '__main__':
    # # Test loading the utility matrix for cf
    # from config import ConfigCf
    #
    # um_movielens = get_um_by_name(ConfigCf, "movielens")
    # um_lecture = get_um_by_name(ConfigCf, "lecture_1")

    # Test the TF dataset loading
    from config import ConfigLf

    ratings_tf, user_ids_voc, movie_ids_voc = load_movielens_tf(ConfigLf)

    # Split the dataset
    train_ds, valid_ds, test_ds = split_train_valid_test_tf(ratings_tf, ConfigLf.split_ratios)
    print_sample_of_tf_dataset(train_ds, "Training:") # 从数据集 train_ds 中打印一部分样本
    print_sample_of_tf_dataset(valid_ds, "Validation:")
    print_sample_of_tf_dataset(test_ds, "Test:")