from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDRec(nn.Module):
    def __init__(self, args, alpha, beta, τ):
        super(CDRec, self).__init__()
        self.args = args
        self.τ = τ
        self.embeds = torch.nn.Embedding(num_embeddings=args.n_node, embedding_dim=args.embedding_dim)
        self.pre_embeds = None
        self.alpha = torch.tensor(alpha.todense(), device=self.args.device)
        self.beta = torch.tensor(beta.todense(), device=self.args.device)
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.embeds.weight, std=1e-4)

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def cal_loss(self, input, nodes, pos_items, pos_weight, neg_weight):
        input = torch.nn.functional.normalize(input, p=2, dim=-1)
        node_embeds = input[nodes]
        pos_embeds = input[pos_items]
        con_pos_score = torch.multiply(node_embeds, pos_embeds).sum(-1)
        con_ttl_score = torch.mm(node_embeds, input.T)

        con_pos_score = pos_weight * torch.exp(con_pos_score / self.τ)
        con_ttl_score = (neg_weight * torch.exp(con_ttl_score / self.τ)).sum(-1)
        loss = -torch.log(con_pos_score / con_ttl_score)

        return loss.sum()

    def forward(self, users, pos_items):
        input = self.embeds.weight
        if type(self.pre_embeds) != type(None):
            input = torch.hstack((input, self.pre_embeds))

        pos_weight = self.alpha[users, pos_items]
        neg_weight = self.beta[users]
        loss = self.cal_loss(input, users, pos_items, pos_weight, neg_weight)
        loss += self.args.γ * self.norm_loss()
        return loss

    def test_foward(self, users):
        input = self.embeds.weight
        if type(self.pre_embeds) != type(None):
            input = torch.hstack((input, self.pre_embeds))
        items = torch.arange(self.args.n_tail).to(self.args.device) + self.args.n_head
        user_embeds = input[users]
        item_embeds = input[items]
        scores = user_embeds.mm(item_embeds.t())
        return scores




