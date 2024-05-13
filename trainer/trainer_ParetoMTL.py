import logging
import warnings
import pickle

import numpy as np
from sklearn import metrics
from torch.backends import cudnn
import torch
from torch import nn
from torch.autograd import Variable

from model.GMKT import GMKT
from model.TAMKOT import TAMKOT
from dataloader.GMKT_dataloader import GMKT_DataLoader
from dataloader.TAMKOT_dataloader import TAMKOT_DataLoader

from utils.min_norm_solvers import MinNormSolver
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")
cudnn.benchmark = True



class trainer_ParetoMTL(object):
    def __init__(self, config, data,  ref_vec, pref_idx):
        super(trainer_ParetoMTL, self).__init__()
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.metric = config.metric

        self.mode = config.mode
        self.manual_seed = config.seed
        self.device = torch.device("cpu")

        self.current_epoch = 1
        self.current_iteration = 1

        self.weights = []
        self.task_train_losses = []
        self.task_test_losses = []
        self.train_evals = []
        self.test_evals = []

        self.ref_vec = ref_vec
        self.pref_idx = pref_idx

        self.ref_vec = self.ref_vec.to(self.device)
        # self.pref_idx= self.pref_idx.to(self.device)

        if self.config.model_name == 'GMKT':
            self.data_loader = GMKT_DataLoader(config, data)
            self.model = GMKT(config, self.data_loader.q_q_neighbors, self.data_loader.q_l_neighbors,
                                            self.data_loader.l_q_neighbors, self.data_loader.l_l_neighbors)

        elif self.config.model_name == 'TAMKOT':
            self.data_loader = TAMKOT_DataLoader(config, data)
            self.model = TAMKOT(config)

        print('==>>> total trainning batch number: {}'.format(len(self.data_loader.train_loader)))
        print('==>>> total testing batch number: {}'.format(len(self.data_loader.test_loader)))

        # self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.BCELoss(reduction='sum')
        # self.criterion = nn.BCELoss(reduction='mean')
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.config.learning_rate,
                                              betas=(config.beta1, config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15, 20, 30, 40, 50, 60], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20, 30, 40, 50, 60], gamma=0.1)

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")



    def train(self):
        # print the current preference vector
        print('Preference Vector ({}/{}):'.format(self.pref_idx + 1, self.config.num_pref))
        print(self.ref_vec[self.pref_idx].cpu().numpy())

        #find the initial solution, stop early once a feasible solution is found, usually can be found with a few steps
        self.train_find_init_solu(2)

        # run niter epochs of ParetoMTL
        for epoch in range(1, self.config.max_epoch + 1):
            # print("=" * 50 + "Epoch {}".format(epoch) + "=" * 50)
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

        if self.config.model_name == 'GMKT':
            torch.save(self.model.state_dict(),
                       'saved_model/{}/{}/np_{}_pi_{}_sl_{}_eq_{}_ea_{}_el_{}_nc_{}_kd_{}_vd_{}_sd_{}_lq_{}_ll_{}_wd_{}_fold_{}.pkl'.format(
                           self.config.data_name,
                           self.config.model_name,
                           self.config.num_pref,
                           self.pref_idx,
                           self.config.max_seq_len,
                           self.config.embedding_size_q,
                           self.config.embedding_size_a,
                           self.config.embedding_size_l,
                           self.config.num_concepts,
                           self.config.key_dim,
                           self.config.value_dim,
                           self.config.summary_dim,
                           self.config.lambda_q,
                           self.config.lambda_l,
                           self.config.weight_decay,
                           self.config.fold))
        elif self.config.model_name == 'TAMKOT':
            torch.save(self.model.state_dict(),
                       'saved_model/{}/{}/np_{}_pi_{}_sl_{}_eq_{}_ea_{}_el_{}_hs_{}_wd_{}_fold_{}.pkl'.format(
                           self.config.data_name,
                           self.config.model_name,
                           self.config.num_pref,
                           self.pref_idx,
                           self.config.max_seq_len,
                           self.config.embedding_size_q,
                           self.config.embedding_size_a,
                           self.config.embedding_size_l,
                           self.config.hidden_size,
                           self.config.weight_decay,
                           self.config.fold))



    def train_find_init_solu(self, init_epochs):
        # run at most 2 epochs to find the initial solution
        # stop early once a feasible solution is found
        # usually can be found with a few steps
        for t in range(init_epochs):

            self.model.train()
            for batch_idx, data in enumerate(self.data_loader.train_loader):
                q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                self.optimizer.zero_grad()
                output, output_type = self.model(q_list, a_list, l_list, d_list)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.criterion(output_type.float(), label_type.float())

                task_loss = torch.stack([loss_q, loss_type])

                # obtain and store the gradient value
                grads = {}
                losses_vec = []

                for i in range(self.config.n_tasks):
                    # self.optimizer.zero_grad()
                    # task_loss = model(X, ts)
                    losses_vec.append(task_loss[i].data)

                    task_loss[i].backward(retain_graph=True)

                    grads[i] = []

                    # can use scalable method proposed in the MOO-MTL paper for large scale problem
                    # but we keep use the gradient of all parameters in this experiment
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
                grads = torch.stack(grads_list)

                # calculate the weights
                losses_vec = torch.stack(losses_vec)
                # losses_vec = losses_vec.to(self.device)
                flag, weight_vec = self.get_d_paretomtl_init(grads, losses_vec, self.ref_vec, self.pref_idx)

                # print(flag, weight_vec)

                # early stop once a feasible solution is obtained
                if flag == True:
                    print("fealsible solution is obtained.")
                    break

                # optimization step
                # self.optimizer.zero_grad()
                for i in range(self.config.n_tasks):
                    # task_loss = model(X, ts)
                    self.optimizer.zero_grad()
                    output, output_type = self.model(q_list, a_list, l_list, d_list)

                    label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                    label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                    output = torch.masked_select(output, target_masks_list[:, 2:])
                    loss_q = self.criterion(output.float(), label.float())

                    output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                    loss_type = self.criterion(output_type.float(), label_type.float())

                    task_loss = torch.stack([loss_q, loss_type])

                    if i == 0:
                        loss_total = weight_vec[i] * task_loss[i]
                    else:
                        loss_total = loss_total + weight_vec[i] * task_loss[i]

                loss_total.backward()
                self.optimizer.step()

            else:
                # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

        # print('')



    def train_one_epoch(self):
        self.model.train()
        for batch_idx, data in enumerate(self.data_loader.train_loader):
            q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
            q_list = q_list.to(self.device)
            a_list = a_list.to(self.device)
            l_list = l_list.to(self.device)
            d_list = d_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            target_masks_list = target_masks_list.to(self.device)
            target_masks_l_list = target_masks_l_list.to(self.device)

            self.optimizer.zero_grad()
            output, output_type = self.model(q_list, a_list, l_list, d_list)

            label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
            label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

            output = torch.masked_select(output, target_masks_list[:, 2:])
            loss_q = self.criterion(output.float(), label.float())

            output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
            loss_type = self.criterion(output_type.float(), label_type.float())

            task_loss = torch.stack([loss_q, loss_type])

            # obtain and store the gradient
            grads = {}
            losses_vec = []

            for i in range(self.config.n_tasks):
                # self.optimizer.zero_grad()
                # task_loss = model(X, ts)
                losses_vec.append(task_loss[i].data)

                task_loss[i].backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                grads[i] = []

                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in self.model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = self.get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)

            normalize_coeff = self.config.n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff

            # optimization step
            self.optimizer.zero_grad()
            for i in range(len(task_loss)):
                # task_loss = model(X, ts)
                output, output_type = self.model(q_list, a_list, l_list, d_list)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.criterion(output_type.float(), label_type.float())

                task_loss = torch.stack([loss_q, loss_type])
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]

            self.weight_vec = weight_vec
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()


    def validate(self):
        self.train_loss = 0
        self.train_loss_type = 0
        train_elements = 0
        train_elements_type = 0

        self.test_loss = 0
        self.test_loss_type = 0
        test_elements = 0
        test_elements_type = 0

        self.model.eval()
        with torch.no_grad():
            self.train_output_all = []
            self.train_output_type_all = []
            self.train_label_all = []
            self.train_label_type_all = []

            self.test_output_all = []
            self.test_output_type_all = []
            self.test_label_all = []
            self.test_label_type_all = []

            for batch_idx, data in enumerate(self.data_loader.train_loader):
                q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                self.optimizer.zero_grad()
                output, output_type = self.model(q_list, a_list, l_list, d_list)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.criterion(output_type.float(), label_type.float())

                self.train_loss += loss_q.item()
                train_elements += target_masks_list[:, 2:].int().sum()
                self.train_loss_type += loss_type.item()
                train_elements_type += (target_masks_list + target_masks_l_list)[:, 2:].int().sum()

                self.train_output_all.extend(output.tolist())
                self.train_output_type_all.extend(output_type.tolist())
                self.train_label_all.extend(label.tolist())
                self.train_label_type_all.extend(label_type.tolist())


            for batch_idx, data in enumerate(self.data_loader.test_loader):
                q_list, a_list, l_list, d_list, target_answers_list, target_masks_list, target_masks_l_list = data
                q_list = q_list.to(self.device)
                a_list = a_list.to(self.device)
                l_list = l_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                target_masks_list = target_masks_list.to(self.device)
                target_masks_l_list = target_masks_l_list.to(self.device)

                self.optimizer.zero_grad()
                output, output_type = self.model(q_list, a_list, l_list, d_list)

                label = torch.masked_select(target_answers_list[:, 2:], target_masks_list[:, 2:])
                label_type = torch.masked_select(d_list[:, 2:], (target_masks_list + target_masks_l_list)[:, 2:])

                output = torch.masked_select(output, target_masks_list[:, 2:])
                loss_q = self.criterion(output.float(), label.float())

                output_type = torch.masked_select(output_type, (target_masks_list + target_masks_l_list)[:, 2:])
                loss_type = self.criterion(output_type.float(), label_type.float())

                self.test_loss += loss_q.item()
                test_elements += target_masks_list[:, 2:].int().sum()
                self.test_loss_type += loss_type.item()
                test_elements_type += (target_masks_list + target_masks_l_list)[:, 2:].int().sum()

                self.test_output_all.extend(output.tolist())
                self.test_output_type_all.extend(output_type.tolist())
                self.test_label_all.extend(label.tolist())
                self.test_label_type_all.extend(label_type.tolist())

            # train_acc = np.stack(
            #     [1.0 * correct1_train / len(train_loader.dataset), 1.0 * correct2_train / len(train_loader.dataset)])

            self.train_output_all = np.array(self.train_output_all).squeeze()
            self.train_label_all = np.array(self.train_label_all).squeeze()
            self.train_output_type_all = np.array(self.train_output_type_all).squeeze()
            self.train_label_type_all = np.array(self.train_label_type_all).squeeze()
            self.test_output_all = np.array(self.test_output_all).squeeze()
            self.test_label_all = np.array(self.test_label_all).squeeze()
            self.test_output_type_all = np.array(self.test_output_type_all).squeeze()
            self.test_label_type_all = np.array(self.test_label_type_all).squeeze()

            if self.metric == "rmse":
                train_evals = np.stack(
                    [np.sqrt(metrics.mean_squared_error(self.train_label_all, self.train_output_all)),
                     metrics.roc_auc_score(self.train_label_type_all, self.train_output_type_all)])
                test_evals = np.stack(
                    [np.sqrt(metrics.mean_squared_error(self.test_label_all, self.test_output_all)),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all)])
            elif self.metric == "auc":
                train_evals = np.stack(
                    [metrics.roc_auc_score(self.train_label_all, self.train_output_all),
                     metrics.roc_auc_score(self.train_label_type_all, self.train_output_type_all)])
                test_evals = np.stack(
                    [metrics.roc_auc_score(self.test_label_all, self.test_output_all),
                     metrics.roc_auc_score(self.test_label_type_all, self.test_output_type_all)])

            # total_train_loss = torch.stack(total_train_loss)
            average_train_loss = torch.stack([self.train_loss / train_elements, self.train_loss_type / train_elements_type])
            average_test_loss = torch.stack([self.test_loss / test_elements, self.test_loss_type / test_elements_type])

        # record and print
        # if torch.cuda.is_available():

        # task_train_losses.append(average_train_loss.data.cpu().numpy())
        self.task_train_losses.append(average_train_loss.data.cpu().numpy())
        self.train_evals.append(train_evals)
        self.task_test_losses.append(average_test_loss.data.cpu().numpy())
        self.test_evals.append(test_evals)

        # weights.append(weight_vec.cpu().numpy())
        self.weights.append(self.weight_vec.cpu().numpy())

        print('{}/{}: weights={}, train_loss={}, train_acc={}, test_loss={}, test_acc={}'.format(
            self.current_epoch, self.config.max_epoch, self.weights[-1], self.task_train_losses[-1], self.train_evals[-1],
            self.task_test_losses[-1], self.test_evals[-1]))

        # self.scheduler.step()
        self.scheduler.step(np.sum(self.task_train_losses[-1]))

    def get_d_paretomtl_init(self, grads, value, ref_vecs, i):
        """
        calculate the gradient direction for ParetoMTL initialization
        """

        flag = False
        nobj = value.shape

        # check active constraints
        current_weight = ref_vecs[i]
        rest_weights = ref_vecs
        w = rest_weights - current_weight

        w = w.to(self.device)
        value = value.to(self.device)

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            flag = True
            return flag, torch.zeros(nobj)
        if torch.sum(idx) == 1:
            # sol = torch.ones(1).cuda().float()
            sol = torch.ones(1).float()
        else:
            vec = torch.matmul(w[idx], grads)
            sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        weight0 = torch.sum(torch.stack([sol[j] * w[idx][j, 0] for j in torch.arange(0, torch.sum(idx))]))
        weight1 = torch.sum(torch.stack([sol[j] * w[idx][j, 1] for j in torch.arange(0, torch.sum(idx))]))
        weight = torch.stack([weight0, weight1])

        return flag, weight

    def get_d_paretomtl(self, grads, value, ref_vecs, i):
        """ calculate the gradient direction for ParetoMTL """

        # check active constraints
        current_weight = ref_vecs[i]
        rest_weights = ref_vecs
        w = rest_weights - current_weight

        w = w.to(self.device)
        value = value.to(self.device)

        gx = torch.matmul(w, value / torch.norm(value))
        idx = gx > 0

        # calculate the descent direction
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            # return torch.tensor(sol).cuda().float()
            return torch.tensor(sol).float()

        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

        weight0 = sol[0] + torch.sum(
            torch.stack([sol[j] * w[idx][j - 2, 0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(
            torch.stack([sol[j] * w[idx][j - 2, 1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        weight = torch.stack([weight0, weight1])

        return weight



