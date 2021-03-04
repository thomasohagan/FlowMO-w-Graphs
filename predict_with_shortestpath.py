# Author: Ryan-Rhys Griffiths
"""
Script for training a model to predict properties in the photoswitch dataset using Gaussian Process Regression.
"""

import argparse

import tensorflow as tf
import networkx as nx
import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pysmiles import read_smiles

from GP.kernels import Shortest_Path
from property_prediction.data_utils import TaskDataLoader


def main(path, task, n_trials, test_set_size, use_rmse_conf):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    :param use_rmse_conf: bool specifying whether to compute the rmse confidence-error curves or the mae confidence-
    error curves. True is the option for rmse.
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    mae_list = []

    # We pre-allocate arrays for plotting confidence-error curves

    _, _, _, y_test = train_test_split(smiles_list, y, test_size=test_set_size)  # To get test set size

    # Photoswitch dataset requires 80/20 splitting. Other datasets are 80/10/10.

    if task != 'Photoswitch':
        split_in_two = int(len(y_test)/2)
        n_test = split_in_two
    else:
        n_test = len(y_test)

    rmse_confidence_list = np.zeros((n_trials, n_test))
    mae_confidence_list = np.zeros((n_trials, n_test))

    print('\nBeginning training loop...')

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(smiles_list, y, test_size=test_set_size, random_state=i)

        if task != 'Photoswitch':

            # Artificially create a 80/10/10 train/validation/test split discarding the validation set.
            split_in_two = int(len(y_test)/2)
            X_test = X_test[0:split_in_two]
            y_test = y_test[0:split_in_two]

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)


            # Convert SMILES of training data to graphs
            graphs_train = [read_smiles(smiles) for smiles in X_train]
            # Convert graphs to adjacency matrices
            X_train = [nx.to_numpy_array(graph) for graph in graphs_train]
            # Pads all matrices to 55 x 55
            for i in range(len(X_train)):
                size = X_train[i].shape[0]
                diff = 55 - size
                X_train[i] = np.pad(X_train[i], [(0,diff), (0,diff)], mode='constant', constant_values=0)

            X_train = np.concatenate(X_train)
            X_train = np.asarray(X_train)
            X_train = X_train.astype(np.float64)


            graphs_test = [read_smiles(smiles) for smiles in X_test]
            X_test = [nx.to_numpy_array(graph) for graph in graphs_test]
            for i in range(len(X_test)):
                size = X_test[i].shape[0]
                diff = 55 - size
                X_test[i] = np.pad(X_test[i], [(0, diff), (0, diff)], mode='constant', constant_values=0)
            X_test = np.concatenate(X_test)
            X_test = np.asarray(X_test)
            X_test = X_test.astype(np.float64)

            k = Shortest_Path()
            m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

            # Optimise the kernel variance and noise level by the marginal likelihood

            opt = gpflow.optimizers.Scipy()
            opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
            print_summary(m)

            # mean and variance GP prediction

            y_pred, y_var = m.predict_f(X_test)

            # Compute scores for confidence curve plotting.

            ranked_confidence_list = np.argsort(y_var, axis=0).flatten()

            for k in range(len(y_test)):

                # Construct the RMSE error for each level of confidence

                conf = ranked_confidence_list[0:k+1]
                rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
                rmse_confidence_list[i, k] = rmse

                # Construct the MAE error for each level of confidence

                mae = mean_absolute_error(y_test[conf], y_pred[conf])
                mae_confidence_list[i, k] = mae

            # Output Standardised RMSE and RMSE on Train Set

            y_pred_train, _ = m.predict_f(X_train)
            train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
            print("Train RMSE: {:.3f}".format(train_rmse))

            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))
            print("MAE: {:.3f}".format(mae))

            r2_list.append(score)
            rmse_list.append(rmse)
            mae_list.append(mae)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))

    # Plot confidence-error curves

    confidence_percentiles = np.arange(1e-14, 100, 100/len(y_test))  # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29

    if use_rmse_conf:

        rmse_mean = np.mean(rmse_confidence_list, axis=0)
        rmse_std = np.std(rmse_confidence_list, axis=0)

        # We flip because we want the most confident predictions on the right-hand side of the plot

        rmse_mean = np.flip(rmse_mean)
        rmse_std = np.flip(rmse_std)

        # One-sigma error bars

        lower = rmse_mean - rmse_std
        upper = rmse_mean + rmse_std

        plt.plot(confidence_percentiles, rmse_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('RMSE')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100*((len(y_test) - 1) / len(y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(task + '/results/shortest_path/{}_confidence_curve_rmse.png'.format("Graph"))
        plt.show()

    else:

        # We plot the Mean-absolute error confidence-error curves

        mae_mean = np.mean(mae_confidence_list, axis=0)
        mae_std = np.std(mae_confidence_list, axis=0)

        mae_mean = np.flip(mae_mean)
        mae_std = np.flip(mae_std)

        lower = mae_mean - mae_std
        upper = mae_mean + mae_std

        plt.plot(confidence_percentiles, mae_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('MAE')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(task + '/results/shortest_path/{}_confidence_curve_mae.png'.format("Graph"))
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../datasets/ESOL.csv',
                        help='Path to the csv file for the task.')
    parser.add_argument('-t', '--task', type=str, default='ESOL',
                        help='str specifying the task. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-n', '--n_trials', type=int, default=1,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-rms', '--use_rmse_conf', type=bool, default=True,
                        help='bool specifying whether to compute the rmse confidence-error curves or the mae '
                             'confidence-error curves. True is the option for rmse.')

    args = parser.parse_args()

    main(args.path, args.task, args.n_trials, args.test_set_size, args.use_rmse_conf)
