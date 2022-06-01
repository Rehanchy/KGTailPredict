import torch
import numpy as np
import torch.nn.functional as function
import random

# a simple loss function
class LossRate_H(torch.nn.Module):
    def __init__(self, margin: float=0.5, C: float= 0.015625, epsilon: float = 0.001):
        super(LossRate_H, self).__init__()
        self.margin = margin
        self.C = C
        self.epsilon = epsilon

    def forward(self, score, neg_score, scale_loss, orthogonal_loss):
        margin_loss = function.relu(self.margin + score - neg_score).mean()
        orthogonal_loss = function.relu(orthogonal_loss - self.epsilon * self.epsilon).mean()
        return margin_loss + self.C * (scale_loss + orthogonal_loss)

class TransH(torch.nn.Module):
    def __init__(self, entity_word2index: dict,
                       relation_word2index: dict,
                       embed_dimension: int = 100,
                       pre_embedded_e: np.ndarray = None,
                       pre_embedded_r: np.ndarray = None):
        super(TransH, self).__init__()
        self.entity_word2index = entity_word2index
        self.relation_word2index = relation_word2index
        entity_word_num = len(self.entity_word2index)
        relation_word_num = len(self.relation_word2index)
        self.embedEntity = torch.nn.Embedding(entity_word_num, embed_dimension)
        self.embedRelation = torch.nn.Embedding(relation_word_num, embed_dimension)
        self.norm_embedding = torch.nn.Embedding(relation_word_num, embed_dimension)
        torch.nn.init.xavier_normal_(self.norm_embedding.weight.data) # provide a hyperface
        if pre_embedded_e is not None and pre_embedded_r is not None:
            print("Pre embedding not None procedure")
            self.embedEntity.weight.data.copy_(torch.from_numpy(pre_embedded_e))
            self.embedRelation.weight.data.copy_(torch.from_numpy(pre_embedded_r))
        else:
            print("Pre embedding None procedure")
            torch.nn.init.xavier_normal_(self.embedEntity.weight.data)
            torch.nn.init.xavier_normal_(self.embedRelation.weight.data)
        self.entity_index2word = {}
        for key, value in self.entity_word2index.items():
            self.entity_index2word[value] = key

    def projection(self, vector, norm):
        norm = function.normalize(norm, 2, dim = -1)
        return vector - torch.sum(norm * vector, dim = -1, keepdim = True) * norm  # hr = hr - wThw

    def normalizor(self):
        self.embedEntity.weight.data = function.normalize(self.embedEntity.weight.data, 2, -1)
        self.norm_embedding.weight.data = function.normalize(self.norm_embedding.weight.data, 2, -1)

    def forward(self, data):
        h, basic_r, t = data
        h = h.flatten()
        basic_r = basic_r.flatten()
        t = t.flatten()
        h = torch.LongTensor([[self.entity_word2index[word.item()]] for word in h]).to(h.device)
        basic_r = torch.LongTensor([[self.relation_word2index[word.item()]] for word in basic_r]).to(h.device)
        t = torch.LongTensor([[self.entity_word2index[word.item()]] for word in t]).to(h.device)
        h = self.embedEntity(h)
        t = self.embedEntity(t)
        r = self.embedRelation(basic_r)
        r_norm = self.norm_embedding(basic_r)
        h_r = self.projection(h, r_norm)
        t_r = self.projection(t, r_norm)
        neg_sample_index = torch.LongTensor([[random.randint(0, len(self.entity_word2index) - 1)] for _ in range(len(t))]).to(h.device)   # random tail entity
        neg_sample = self.embedEntity(neg_sample_index)
        neg_sample_r = self.projection(neg_sample, r_norm)
        score = torch.norm((h_r + r) - t_r, p = 2, dim = -1).flatten()
        neg_score = torch.norm((h_r + r) - neg_sample_r, p = 2, dim = -1).flatten()
        scale_loss = function.relu(torch.norm(self.embedEntity.weight.data, p = 2, dim = -1) - 1).mean()
        orthogonal_loss = torch.sum(self.embedRelation.weight.data * self.norm_embedding.weight.data, dim = -1) / torch.norm(self.embedRelation.weight.data, p = 2, dim = -1) 
        return score, neg_score, scale_loss, orthogonal_loss

    def predict(self, data):
        h, basic_r = data
        h = h.flatten()
        basic_r = basic_r.flatten()
        h = torch.LongTensor([[self.entity_word2index[word.item()]] for word in h]).to(h.device)
        basic_r = torch.LongTensor([[self.relation_word2index[word.item()]] for word in basic_r]).to(h.device)
        h = self.embedEntity(h)
        r = self.embedRelation(basic_r)
        r_norm = self.norm_embedding(basic_r)
        h_r = self.projection(h, r_norm)
        predict = h_r + r
        results = {"hit@1": [], "hit@5": [], "hit@10": []}
        for tail_predict, normal_plane in zip(predict, r_norm):
            entity_hyper_embedding = self.embedEntity.weight.data 
            - torch.matmul(torch.sum(self.embedEntity.weight.data * normal_plane, dim = -1, keepdim = True), normal_plane)
            scoreMat = tail_predict - entity_hyper_embedding
            scorelist = torch.norm(scoreMat, p = 2, dim = -1, keepdim = False)
            result_index = torch.topk(scorelist, k = 10, dim = -1, largest = False)[1].tolist()
            result_words = [self.entity_index2word[idx] for idx in result_index]
            results["hit@1"].append(result_words[:1])
            results["hit@5"].append(result_words[:5])
            results["hit@10"].append(result_words[:10])
        return results

    