import torch
import torch.nn as nn

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization


class NewModel(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NewModel, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        pos_item_e = self.item_embedding(pos_item)                # [batch_size, embedding_size]
        neg_item_e = self.item_embedding(neg_item)                # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)          # []

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)            # [batch_size, embedding_size]
        item_e = self.item_embedding(item)            # [batch_size, embedding_size]

        scores = torch.mul(user_e, item_e).sum(dim=1) # [batch_size]

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight                   # [n_items, batch_size]

        scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]

        return scores