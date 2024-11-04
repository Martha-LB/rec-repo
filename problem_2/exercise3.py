import dataclasses
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds


def mse_loss_all_batches(mat_u, mat_v, dataset, batch_size):
    """ Compute mse per batch using vectorized operations
        Returns a list of mse values for all batches as floats
    """
    mse_all_batches = []
    #dataset.batch(size) 将数据集分割成多个批次，没个批次包含size个样本
    #tfds.as_numpy将tensorflow数据集转化为numpy格式
    for record in tfds.as_numpy(dataset.batch(batch_size)):
        mse = mse_loss_one_batch(mat_u, mat_v, record)
        mse_all_batches.append(mse)
    # convert list of arrays to list of floats
    mse_all_batches = list(map(float, mse_all_batches))
    return mse_all_batches



@jax.jit  # Comment out for single-step debugging
def mse_loss_one_batch(mat_u, mat_v, record):
    """This colab experiment motivates the implementation:
    https://colab.research.google.com/drive/1c0LpSndbTJaHVoLTatQCbGhlsWbpgvYh?usp&#x3D;sharing=
    """
    rows, columns, ratings = record["movie_id"], record["user_id"], record["user_rating"]
    estimator = -(mat_u @ mat_v)[(rows, columns)] # rows和columns是数组，分别包含着行和列的所有索引。而(rows,columns)是一组索引值，在矩阵中提取中对应索引位置的值们，并返回一个数组。
    #注意数组estimator中是负数
    #jnp.square对每个误差值进行平方
    square_err = jnp.square(estimator + ratings) #jnp.square是jax库提供的函数，保证这个平方计算可以兼容jax的计算机制
    mse = jnp.mean(square_err)
    return mse


def init_latent_factors(num_users, num_items, num_factors, rng_key):
    """ Initialize latent factors for users and items
    """
    key_u, key_v = jax.random.split(rng_key)
    #jax.random.normal()用于生成一个正态分布随机数的矩阵
    #(num_items, num_factors)是生成的矩阵的形状，表示有num_items行和num_factors列
    matrix_u = jax.random.normal(key_u, (num_items, num_factors))
    print(f"矩阵前几行是：{matrix_u[:5]}")
    matrix_v = jax.random.normal(key_v, (num_factors, num_users))
    return matrix_u, matrix_v


def load_data_and_init_factors(config):
    # load the dataset using TensorFlow Datasets
    import data_util as data

    #data.load_movielens_tf函数用于加载movielens数据集，且数据集格式是tensorflow数据集
    #rating_tf是movielens数据集的tensorflow数据集格式，包含用户ID、电影ID、评分信息
    #user_ids_voc和movie_ids_voc是用户和电影的词汇表
    ratings_tf, user_ids_voc, movie_ids_voc = data.load_movielens_tf(config)
    #user_ids_voc.get_vocabulary()返回一个数组或者列表，返回数据集中所有唯一的用户ID
    num_users = len(user_ids_voc.get_vocabulary())
    num_items = len(movie_ids_voc.get_vocabulary())
    #jax.random.PRNGKey JAX中随机数生成器生成密钥来控制随机性，这里用rng_seed设置随机数种子
    #jax.random.split 分割生成的密钥，将其拆分成两个密钥。后续不同操作使用不用随机性。
    #rng_key_factors初始化潜在因子矩阵的随机性
    rng_key_factors, rng_key_r = jax.random.split(jax.random.PRNGKey(config.rng_seed))

    print(f"密钥rng_key_factor : {rng_key_factors}")

    #num_factors指定了潜在因子的维数（即矩阵的列数）
    #rng_key_factors随机数生成密钥
    matrix_u, matrix_v = init_latent_factors(num_users, num_items, config.num_factors, rng_key_factors)
    return ratings_tf, matrix_u, matrix_v, num_users, num_items


def predict_and_compare(mat_u, mat_v, train_ds, config):
    """ Predict ratings for the test dataset, compare to target ratings
        Returns a list of tuples with (predicted, target) ratings"""

    predictions_and_targets = []
    # Only batch_size=1 is supported for now

    for idx, record in enumerate(tfds.as_numpy(train_ds.batch(1))):
        # Batch sizes > 1 compute too many predictions (all pairs of users and items)
        # i, j, rating = record["user_id"], record["movie_id"], record["user_rating"]
        # rating_pred = jnp.dot(mat_u[i, :], mat_v[:, j])
        i, j, rating = record["movie_id"][0], record["user_id"][0], float(record["user_rating"][0])
        rating_pred = float(jnp.dot(mat_u[i, :], mat_v[:, j]))
        predictions_and_targets.append((rating_pred, rating))
        if idx >= config.num_records_predict_and_compare:
            break
    return predictions_and_targets


def uv_factorization_vec_no_reg(mat_u, mat_v, train_ds, valid_ds, config):
    """ Matrix factorization using SGD without regularization
        Fast vectorized implementation using JAX
    """

    @jax.jit  # Comment out for single-step debugging
    def update_uv(mat_u, mat_v, record, lr):
        #使用jax的jax.value_and_grad函数计算当前批次的损失
        #value_and_grad函数正如其名，它可以计算指定函数（这里是mse_loss_one_batch函数）的输出值以及相对于其参数的梯度
        #argnums=[0,1]指定了要对mse_loss_one_batch函数的第一个和第二个参数求梯度，即mat_u和mat_v
        #最终会返回两个值，分别是loss_value，即该批次的均方误差，以及一个元组(grad_mat_u和grad_mat_v)
        loss_value, grad = jax.value_and_grad(mse_loss_one_batch, argnums=[0, 1])(mat_u, mat_v, record)
        #mat-u和mat_v按梯度下降公式更新如下
        #grad[0]和grad[1]分别是损失对mat_u和mat_v的梯度
        mat_u = mat_u - lr * grad[0]
        mat_v = mat_v - lr * grad[1]
        return mat_u, mat_v, loss_value

    #num_epochs总训练轮数
    for epoch in range(config.num_epochs):
        #lr学习率，优先选择fixed_learning_rate，否则选择动态学习率（指数衰减）
        lr = config.fixed_learning_rate if config.fixed_learning_rate is not None \
            else config.dyn_lr_initial * (config.dyn_lr_decay_rate ** (epoch / config.dyn_lr_steps))
        print(f"In uv_factorization_vec_no_reg, starting epoch {epoch} with lr={lr:.6f}")
        train_loss = []
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training)):
            mat_u, mat_v, loss = update_uv(mat_u, mat_v, record, lr)
            train_loss.append(loss)

        train_loss_mean = jnp.mean(jnp.array(train_loss)) #所有批次损失的均值——平均训练损失
        # Compute loss on the validation set
        valid_loss = mse_loss_all_batches(mat_u, mat_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss)) #平均验证损失
        print(
            f"Epoch {epoch} finished, ave training loss: {train_loss_mean:.6f}, ave validation loss: {valid_loss_mean:.6f}")
    return mat_u, mat_v


def uv_factorization_vec_reg(mat_u, mat_v, train_ds, valid_ds, config):
    """ Matrix factorization using SGD with regularization
        Fast vectorized implementation using JAX
    """

    @jax.jit  # Comment out for single-step debugging
    def update_uv(mat_u, mat_v, record, lr, reg_param):
        #使用jax的jax.value_and_grad函数计算当前批次的损失
        #value_and_grad函数正如其名，它可以计算指定函数（这里是mse_loss_one_batch函数）的输出值以及相对于其参数的梯度
        #argnums=[0,1]指定了要对mse_loss_one_batch函数的第一个和第二个参数求梯度，即mat_u和mat_v
        #最终会返回两个值，分别是loss_value，即该批次的均方误差，以及一个元组(grad_mat_u和grad_mat_v)
        loss_value, grad = jax.value_and_grad(mse_loss_one_batch, argnums=[0, 1])(mat_u, mat_v, record)
        #mat-u和mat_v按梯度下降公式更新如下
        #grad[0]和grad[1]分别是损失对mat_u和mat_v的梯度
        grad_u = grad[0] + reg_param * mat_u
        grad_v = grad[1] + reg_param * mat_v
        mat_u = mat_u - lr * grad_u
        mat_v = mat_v - lr * grad_v
        return mat_u, mat_v, loss_value

    #num_epochs总训练轮数
    for epoch in range(config.num_epochs):
        lr = config.fixed_learning_rate if config.fixed_learning_rate is not None else config.dyn_lr_initial
        reg_param = config.reg_param
        print(f"In uv_factorization_vec_no_reg, starting epoch {epoch} with lr={lr:.6f}")
        train_loss = []
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training)):
            mat_u, mat_v, loss = update_uv(mat_u, mat_v, record, lr, reg_param)
            train_loss.append(loss)

        train_loss_mean = jnp.mean(jnp.array(train_loss)) #所有批次损失的均值——平均训练损失
        # Compute loss on the validation set
        valid_loss = mse_loss_all_batches(mat_u, mat_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss)) #平均验证损失
        print(
            f"Epoch {epoch} finished, ave training loss: {train_loss_mean:.6f}, ave validation loss: {valid_loss_mean:.6f}")
    return mat_u, mat_v


import itertools

def grid_search_uv_factorization(train_ds, valid_ds, matrix_u_init, matrix_v_init, config):
    """ Perform grid search for hyperparameters fixed_learning_rate and reg_param """
    # Define grid for hyperparameters
    learning_rates = jnp.linspace(0.001, 0.1, 5)  # Sample values for learning rates
    reg_params = jnp.linspace(0.01, 0.1, 5)       # Sample values for regularization params

    best_valid_loss = float("inf")
    best_hyperparams = None
    best_matrix_u, best_matrix_v = None, None

    for lr, reg in itertools.product(learning_rates, reg_params):
        # Reset U and V matrices for each grid search trial
        matrix_u, matrix_v = matrix_u_init.copy(), matrix_v_init.copy()

        # Update config with current hyperparameters
        config.fixed_learning_rate = lr
        config.reg_param = reg

        # Run training with regularization
        print(f"Testing lr={lr:.6f}, reg={reg:.6f}")
        matrix_u, matrix_v = uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, valid_ds, config)

        # Compute validation loss
        valid_loss = mse_loss_all_batches(matrix_u, matrix_v, valid_ds, config.batch_size_predict_with_mse)
        valid_loss_mean = jnp.mean(jnp.array(valid_loss))

        # Update best model if validation loss improves
        if valid_loss_mean < best_valid_loss:
            best_valid_loss = valid_loss_mean
            best_hyperparams = (lr, reg)
            best_matrix_u, best_matrix_v = matrix_u, matrix_v

    print(f"Best hyperparameters: lr={best_hyperparams[0]}, reg={best_hyperparams[1]} with validation loss {best_valid_loss:.6f}")
    return best_matrix_u, best_matrix_v, best_hyperparams



@dataclasses.dataclass
class Flags:
    problem_a = False
    problem_b = True


if __name__ == '__main__':
    from config import ConfigLf as config
    import data_util as data

    if Flags.problem_a:
        # Initialize matrices and datasets
        ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
        train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

        def show_metrics_and_examples(message, matrix_u, matrix_v):
            print(message)
            mse_all_batches = mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
            print("MSE examples from predict_with_mse on test_ds")
            print(mse_all_batches[:config.num_predictions_to_show])
            print("Prediction examples (pred, target)")
            predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
            print(predictions_and_targets[:config.num_predictions_to_show])

        # Show initial metrics before optimization
        show_metrics_and_examples("====== Before optimization (Regularized) =====", matrix_u, matrix_v)

        # Run the regularized matrix factorization
        matrix_u, matrix_v = uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, valid_ds, config)

        # Show metrics after optimization
        show_metrics_and_examples("====== After optimization (Regularized) =====", matrix_u, matrix_v)


    if Flags.problem_b:
        ratings_tf, matrix_u, matrix_v, num_users, num_items = load_data_and_init_factors(config)
        train_ds, valid_ds, test_ds = data.split_train_valid_test_tf(ratings_tf, config)

        # Perform grid search to find optimal hyperparameters
        best_matrix_u, best_matrix_v, best_hyperparams = grid_search_uv_factorization(train_ds, valid_ds, matrix_u, matrix_v, config)

        def show_metrics_and_examples(message, matrix_u, matrix_v):
            print(message)
            mse_all_batches = mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
            print("MSE examples from predict_with_mse on test_ds")
            print(mse_all_batches[:config.num_predictions_to_show])
            print("Prediction examples (pred, target)")
            predictions_and_targets = predict_and_compare(matrix_u, matrix_v, test_ds, config)
            print(predictions_and_targets[:config.num_predictions_to_show])

        # Display results for the best regularized model
        show_metrics_and_examples("====== After optimization with best regularized model =====", best_matrix_u, best_matrix_v)

        # Run the non-regularized model for comparison
        matrix_u, matrix_v = uv_factorization_vec_no_reg(matrix_u, matrix_v, train_ds, valid_ds, config)
        show_metrics_and_examples("====== After optimization with non-regularized model =====", matrix_u, matrix_v)

        