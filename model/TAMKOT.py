import torch
import torch.nn as nn
import geotorch


class TAMKOT(nn.Module):
    '''
    Multiview Deep knowledge tracing model
    '''
    def __init__(self, config):
        super(TAMKOT, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric

        self.num_users = config.num_users
        self.num_questions = config.num_items
        self.num_nongradable_items = config.num_nongradable_items
        self.embeding_size_q = config.embedding_size_q
        self.embeding_size_a = config.embedding_size_a
        # self.embeding_size_qa = config.embedding_size_qa
        self.embeding_size_l = config.embedding_size_l
        self.hidden_size = config.hidden_size

        self.init_std = config.init_std

        # initialize embedding layer
        if self.metric == "rmse":
            self.a_embed_matrix = nn.Linear(1, self.embeding_size_a)
        else:
            self.a_embed_matrix = nn.Embedding(3, self.embeding_size_a, padding_idx=2)

        self.q_embed_matrix = nn.Embedding(num_embeddings=self.num_questions + 1,
                                           embedding_dim=self.embeding_size_q,
                                           padding_idx=0)
        self.q_bais = nn.Embedding(num_embeddings=self.num_questions + 1, embedding_dim=1, padding_idx=0)
        # self.qa_embed_matrix = nn.Embedding(num_embeddings=2 * self.num_questions + 1,
        #                                         embedding_dim=self.value_dim,
        #                                         padding_idx=0)


        self.l_embed_matrix = nn.Embedding(num_embeddings=self.num_nongradable_items + 1,
                                           embedding_dim=self.embeding_size_l,
                                           padding_idx=0)

        self.W_iQ = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.hidden_size, bias=True)
        self.W_iL = nn.Linear(self.embeding_size_l, self.hidden_size, bias=True)

        self.W_ihQQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ihLL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.W_ihQL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ihLQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        self.W_gQ = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.hidden_size, bias=True)
        self.W_gL = nn.Linear(self.embeding_size_l, self.hidden_size, bias=True)

        self.W_ghQQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ghLL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.W_ghQL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ghLQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        self.W_fQ = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.hidden_size, bias=True)
        self.W_fL = nn.Linear(self.embeding_size_l, self.hidden_size, bias=True)

        self.W_fhQQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_fhLL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.W_fhQL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_fhLQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        self.W_oQ = nn.Linear(self.embeding_size_q + self.embeding_size_a, self.hidden_size, bias=True)
        self.W_oL = nn.Linear(self.embeding_size_l, self.hidden_size, bias=True)

        self.W_ohQQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ohLL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.W_ohQL = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_ohLQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


        self.W_Qh = nn.Linear(self.hidden_size + self.embeding_size_q, 1, bias=True)
        # self.W_Lh = nn.Linear(self.hidden_size + self.embeding_size_l, 1, bias=True)
        self.W_Lh = nn.Linear(self.hidden_size, 1, bias=True)

        # initialize the activate functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, q_data, a_data, l_data, d_data):
        '''
        get output of the model with size (batch_size, seq_len)
        :param q_data:
        :param a_data:
        :param l_data:
        :return:
        '''

        batch_size, seq_len = q_data.size(0), q_data.size(1)
        # inintial h0
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        m = torch.zeros(batch_size, self.hidden_size).to(self.device)

        q_embed = self.q_embed_matrix(q_data)
        q_biases = self.q_bais(q_data)
        if self.metric == 'rmse':
            a_data = torch.unsqueeze(a_data, dim=2)
            a_embed = self.a_embed_matrix(a_data)
        else:
            a_embed = self.a_embed_matrix(a_data)
        l_embed = self.l_embed_matrix(l_data)

        # split the data seq into chunk and process.py each question sequentially
        sliced_q_embed = torch.chunk(q_embed, seq_len, dim=1)
        sliced_q_biases = torch.chunk(q_biases, seq_len, dim=1)
        sliced_a_embed = torch.chunk(a_embed, seq_len, dim=1)
        sliced_l_embed = torch.chunk(l_embed, seq_len, dim=1)
        sliced_d_data = torch.chunk(d_data, seq_len, dim=1)


        batch_pred_q, batch_pred_type = [], []
        for t in range(1, seq_len-1):
            q = sliced_q_embed[t].squeeze(1)
            a = sliced_a_embed[t].squeeze(1)
            l = sliced_l_embed[t].squeeze(1)
            d_t = sliced_d_data[t]
            d_t_1 = sliced_d_data[t - 1]

            qa = torch.cat([q, a], dim = 1)

            i = self.sigmoid((1 - d_t) * self.W_iQ(qa) + d_t * self.W_iL(l) + ((1 - d_t) * (1 - d_t_1)) * self.W_ihQQ(h)
                             + (d_t * d_t_1) * self.W_ihLL(h) + ((1 - d_t_1) * d_t) * self.W_ihQL(h) + (
                                         d_t_1 * (1 - d_t)) * self.W_ihLQ(h))
            g = self.tanh((1 - d_t) * self.W_gQ(qa) + d_t * self.W_gL(l) + ((1 - d_t) * (1 - d_t_1)) * self.W_ghQQ(h)
                             + (d_t * d_t_1) * self.W_ghLL(h) + ((1 - d_t_1) * d_t) * self.W_ghQL(h) + (
                                         d_t_1 * (1 - d_t)) * self.W_ghLQ(h))
            f = self.sigmoid((1 - d_t) * self.W_fQ(qa) + d_t * self.W_fL(l) + ((1 - d_t) * (1 - d_t_1)) * self.W_fhQQ(h)
                             + (d_t * d_t_1) * self.W_fhLL(h) + ((1 - d_t_1) * d_t) * self.W_fhQL(h) + (
                                         d_t_1 * (1 - d_t)) * self.W_fhLQ(h))
            o = self.sigmoid((1 - d_t) * self.W_oQ(qa) + d_t * self.W_oL(l) + ((1 - d_t) * (1 - d_t_1)) * self.W_ohQQ(h)
                             + (d_t * d_t_1) * self.W_ohLL(h) + ((1 - d_t_1) * d_t) * self.W_ohQL(h) + (
                                         d_t_1 * (1 - d_t)) * self.W_ohLQ(h))

            m = f * m + i * g
            h = o * self.tanh(m)

            batch_sliced_pred_type = self.sigmoid(self.W_Lh(h))
            # batch_sliced_pred_l = torch.gather(batch_sliced_pred_l, 1, next_item_l)
            batch_pred_type.append(batch_sliced_pred_type)

            next_item_q = torch.chunk(q_data, seq_len, dim=1)[t + 1]
            next_item_embed_q = sliced_q_embed[t + 1].squeeze(1)
            next_item_biases_q = sliced_q_biases[t + 1].squeeze(1)
            next_q_h = torch.cat([h, next_item_embed_q], dim=1)

            batch_sliced_pred_q = self.sigmoid(self.W_Qh(next_q_h))
            # batch_sliced_pred_q = self.sigmoid(self.W_Qh(next_q_h) + next_item_biases_q)
            # batch_sliced_pred_q = torch.gather(batch_sliced_pred_q, 1, next_item_q)
            batch_pred_q.append(batch_sliced_pred_q)

            # next_item_l = torch.chunk(l_data, seq_len, dim=1)[t + 1]
            # next_item_embed_l = sliced_l_embed[t + 1].squeeze(1)
            # next_l_h = torch.cat([m, next_item_embed_l], dim=1)



        batch_pred_q = torch.cat(batch_pred_q, dim=-1)
        batch_pred_type = torch.cat(batch_pred_type, dim=-1)
        return batch_pred_q, batch_pred_type
























