from numba import njit, jit
import numpy as np
import math


def _fast_cv(dataset, movie_biases, M, global_mean, folds, lr, reg):
    N = dataset.shape[0]
    nf = M.shape[1]
    fold_size = N // folds
    mse = 0
    mae = 0
    tested = 0
    for i in range(folds):
        test_fold = dataset[i * fold_size : (i + 1) * fold_size]
        train = np.vstack((dataset[: i * fold_size], dataset[(i + 1) * fold_size :]))
        ub, U = _fast_foreign_train(
            200,
            train,
            np.random.normal(0, 0.1),
            movie_biases,
            np.random.normal(0, 0.1, nf),
            M,
            global_mean,
            lr,
            reg,
            False,
            False,
        )
        K = test_fold.shape[0]
        for x in range(K):
            mid, rating = int(test_fold[x, 0]), test_fold[x, 1]
            mb = movie_biases[mid]
            mrow = M[mid]
            predicted = _fast_predict(ub, mb, U, mrow, global_mean, nf)
            err = rating - predicted
            mae += abs(err)
            mse += err**2
            tested += K
    return math.sqrt(mse / tested), mae / tested


@njit
def _fast_predict(user_bias, movie_bias, Urow, Mrow, global_mean, nf):
    baseline = global_mean + user_bias + movie_bias
    dot = 0
    for f in range(nf):
        dot += Urow[f] * Mrow[f]
    return baseline + dot


@njit
def _fast_validation_metrics(ratings, U, M, user_biases, movie_biases, global_mean, nf):
    N = ratings.shape[0]
    rmse = 0
    mse = 0
    mae = 0
    for i in range(N):
        uid, mid, rating = int(ratings[i, 0]), int(ratings[i, 1]), ratings[i, 2]
        ub = user_biases[uid]
        mb = movie_biases[mid]
        urow = U[uid]
        mrow = M[mid]
        predicted = _fast_predict(ub, mb, urow, mrow, global_mean, nf)
        err = rating - predicted
        mse += err**2
        mae += abs(err)
    mse = mse / N
    mae = mae / N
    return math.sqrt(mse), mae, mse


@njit
def _fast_train(
    epochs,
    trainset,
    user_bias,
    movie_bias,
    U,
    M,
    global_mean,
    lr,
    reg,
    shuffle,
    verbose,
):
    np.random.shuffle(trainset)
    nf = M.shape[1]
    N = trainset.shape[0]
    for epoch in range(epochs):
        if verbose:
            print("epoch", epoch)
        if shuffle:
            np.random.shuffle(trainset)
        for i in range(N):
            uid, mid, rating = int(trainset[i, 0]), int(trainset[i, 1]), trainset[i, 2]
            pred = _fast_predict(
                user_bias[uid], movie_bias[mid], U[uid], M[mid], global_mean, nf
            )
            err = rating - pred

            # Update biases
            user_bias[uid] += lr * (err - reg * user_bias[uid])
            movie_bias[mid] += lr * (err - reg * movie_bias[mid])

            # Update latent factors
            for f in range(nf):
                puf = U[uid, f]
                qmf = M[mid, f]

                U[uid, f] += lr * (err * qmf - reg * puf)
                M[mid, f] += lr * (err * puf - reg * qmf)

    return U, M, user_bias, movie_bias


@njit
def _fast_foreign_validation(ratings, U, M, user_bias, movie_biases, global_mean, nf):
    N = ratings.shape[0]
    rmse = 0
    mse = 0
    mae = 0
    for i in range(N):
        mid, rating = int(ratings[i, 0]), ratings[i, 1]
        mb = movie_biases[mid]
        mrow = M[mid]
        predicted = _fast_predict(user_bias, mb, U, mrow, global_mean, nf)
        err = rating - predicted
        mse += err**2
        mae += abs(err)
    mse = mse / N
    mae = mae / N
    return math.sqrt(mse), mae, mse


@njit
def _fast_foreign_train(
    epochs,
    trainset,
    user_bias,
    movie_biases,
    U,
    M,
    global_mean,
    lr,
    reg,
    shuffle,
    verbose,
):
    nf = M.shape[1]
    N = trainset.shape[0]
    for epoch in range(epochs):
        if verbose:
            print("epoch", epoch)
        if shuffle:
            np.random.shuffle(trainset)
        for i in range(N):
            mid, rating = int(trainset[i, 0]), trainset[i, 1]
            pred = _fast_predict(
                user_bias, movie_biases[mid], U, M[mid], global_mean, nf
            )
            err = rating - pred

            # Update bias
            user_bias += lr * (err - reg * user_bias)

            # Update latent factors
            for f in range(nf):
                puf = U[f]
                qmf = M[mid, f]

                U[f] += lr * (err * qmf - reg * puf)

    return user_bias, U
