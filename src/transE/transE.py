import torch
import random
import numpy as np
import torch.nn.functional as function

# a simple loss function
class LossRate_E(torch.nn.Module):
    def __init__(self, constant: float=1.0):
        super(LossRate_E, self).__init__()
        self.constant = constant

    def forward(self, score, neg_score):
        score = torch.norm(score, 2, -1).flatten()
        neg_score = torch.norm(neg_score, 2, -1).flatten()
        loss = score - neg_score + self.constant
        return function.relu(loss).mean() # get mean value

# transE model
class TransE(torch.nn.Module):
    def __init__(self, 
                        entity_word2index: dict,
                        relation_word2index: dict,
                        embed_dimension: int = 100,
                        pre_embedded_e: np.ndarray = None,
                        pre_embedded_r: np.ndarray = None):
        super(TransE, self).__init__()
        # get the dict about entity - index or relation -index
        self.entity_word2index = entity_word2index
        self.relation_word2index = relation_word2index
        entity_word_num = len(self.entity_word2index)
        relation_word_num = len(self.relation_word2index)
        # use dimension and num we now to init embedding procedure
        self.embedEntity = torch.nn.Embedding(entity_word_num, embed_dimension)
        self.embedRelation = torch.nn.Embedding(relation_word_num, embed_dimension)
        if pre_embedded_e is not None and pre_embedded_r is not None:
            print("Pre embedding not None procedure")
            self.embedEntity.weight.data.copy_(torch.from_numpy(pre_embedded_e))
            self.embedRelation.weight.data.copy_(torch.from_numpy(pre_embedded_r))
        else:
            print("Pre embedding None procedure")
            torch.nn.init.xavier_normal_(self.embedEntity.weight.data)
            torch.nn.init.xavier_normal_(self.embedRelation.weight.data)
        # 交给神经网络，我们只需要定义我们想要的维度，比如100，然后通过神经网络去学习它的每一个属性的大小，而我们并不用关心到底这个属性代表着什么
        self.embedRelation.weight.data = function.normalize(self.embedRelation.weight.data, 2, -1) # get module
        self.entity_index2word = {}
        # get what index means what word, for prediction
        for key, value in self.entity_word2index.items():
            self.entity_index2word[value] = key

    def normalizor(self):
        self.embedEntity.weight.data = function.normalize(self.embedEntity.weight.data, 2, -1)
    
    def forward(self, data):
        h, r, t = data
        h = h.flatten()
        r = r.flatten()
        t = t.flatten()
        h = torch.LongTensor([[self.entity_word2index[word.item()]] for word in h]).to(h.device)
        r = torch.LongTensor([[self.relation_word2index[word.item()]] for word in r]).to(h.device)
        t = torch.LongTensor([[self.entity_word2index[word.item()]] for word in t]).to(h.device) # get index to search in embedding
        h = self.embedEntity(h)
        t = self.embedEntity(t)
        r = self.embedRelation(r)
        neg_sample_index = torch.LongTensor([[random.randint(0, len(self.entity_word2index) - 1)] for _ in range(len(t))]).to(h.device)   # random tail entity
        neg_sample_score = self.embedEntity(neg_sample_index)
        score = (h + r) - t
        neg_score = (h + r) - neg_sample_score
        return score, neg_score

    def predict(self, data):
        h, r = data
        h = h.flatten()
        r = r.flatten()
        h = torch.LongTensor([[self.entity_word2index[word.item()]] for word in h]).to(h.device)
        r = torch.LongTensor([[self.relation_word2index[word.item()]] for word in r]).to(h.device)
        h = self.embedEntity(h)
        r = self.embedRelation(r)
        predict = h + r
        results = {"hit@1": [], "hit@5": [], "hit@10": []}
        for tail_vec in predict:
            scoreMat = tail_vec - self.embedEntity.weight.data
            scorelist = torch.norm(scoreMat, p = 2, dim = -1, keepdim = False)
            result_index = torch.topk(scorelist, k = 10, dim = -1, largest = False)[1].tolist() # get predicted index
            result_words = [self.entity_index2word[index] for index in result_index]
            results["hit@1"].append(result_words[:1])
            results["hit@5"].append(result_words[:5])
            results["hit@10"].append(result_words[:10])
        return results

    