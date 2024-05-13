import torch
import numpy as np
from easydict import EasyDict
import pickle
from trainer.trainer_ParetoMTL import trainer_ParetoMTL


def circle_points(r, n):
    """
    generate evenly distributed unit divide vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def paretoMLT_exp(config):

    config = EasyDict(config)

    ref_vec = torch.tensor(circle_points([1], [config.num_pref])[0]).float()

    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']

    print(config)

    # for i in range(config.num_pref):
    for i in range(2,3): #3th is the middle preference vector
        pref_idx = i
        exp_trainner = trainer_ParetoMTL(config, data, ref_vec, pref_idx)
        exp_trainner.train()



def pareto_GMKT():
    config = {
        "data_name": 'ednet',
        "model_name": 'GMKT',  #it could be TAMKOT
        # "model_name": 'TAMKOT',

        "mode": 'test',
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "n_tasks": 2,
        "num_pref": 5, #number of dividing vector

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max seq length for model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 60,
        "validation_split": 0.2,

        "embedding_size_q": 32,
        "embedding_size_a": 32,
        "embedding_size_l": 32,
        "num_concepts": 8,
        "hidden_size": 16,  # hidden state size

        "key_dim": 32,
        "value_dim": 32,
        "summary_dim": 32,
        # 'weight_type': 0.1,
        "lambda_q": 0.5,
        "lambda_l": 0.5,


        "init_std": 0.2,
        "max_grad_norm": 10,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.05,
    }
    paretoMLT_exp(config)



def pareto_TAMKOT():
    config = {
        "data_name": 'ednet',
        "model_name": 'TAMKOT',

        "mode": 'test',
        "fold": 1,
        "metric": 'auc',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "n_tasks": 2,
        "num_pref": 5, #number of dividing vector

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max seq length for model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 60,
        "validation_split": 0.2,

        "embedding_size_q": 32,
        "embedding_size_a": 32,
        "embedding_size_l": 32,
        "hidden_size": 16,  # hidden state size

        "init_std": 0.2,
        "max_grad_norm": 10,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.05,
    }
    paretoMLT_exp(config)


if __name__ == '__main__':
    # pareto_GMKT()
    pareto_TAMKOT()
