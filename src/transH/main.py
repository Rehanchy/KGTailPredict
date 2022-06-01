import os
import tqdm
import torch
import pickle
import torch.utils.data
from wordEmbedding import WordEmbedding
from transH import LossRate_H, TransH
from tripletGens import TripletDataset

# modify your data here
epoch_range = 10
dimension = 100
batch_size = 1000
learning_rate = 0.0001
entity_embedding_save_str = "./output/entity_embedded.bin"
relation_embedding_save_str = "./output/relation_embedded.pkl"
train_data_str = "./data/train.txt"
dev_data_str = "./data/dev.txt"
train_model_save_str = "./output/transHtraine2.bin"

def train():
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    if not os.path.isfile(entity_embedding_save_str) or not os.path.isfile(relation_embedding_save_str):
        WordEmbedding("./data/entity_with_text.txt",
                       "./data/relation_with_text.txt",
                       "./data/train.txt",
                       "./data/test.txt",
                       vector_size = dimension)
    entity_pre_embedded = pickle.load(open(entity_embedding_save_str, "rb"))
    relation_pre_embedded = pickle.load(open(relation_embedding_save_str, "rb"))
    best_score = 0.0
    device = "cuda:0"
    if not os.path.isfile(train_model_save_str):
        model = TransH(entity_pre_embedded["word2index"],
                        relation_pre_embedded["rel2idx"],
                        embed_dimension = dimension,
                        pre_embedded_e = entity_pre_embedded["embedded"],
                        pre_embedded_r = relation_pre_embedded["embedding"]).to(device) 
    else:
        checkpoint = torch.load(train_model_save_str, map_location = "cpu")
        entity_word2index = checkpoint["entity_word2index"]
        relation_word2index = checkpoint["relation_word2index"]
        model = TransH(entity_word2index,
                        relation_word2index,
                        embed_dimension = dimension)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        best_score = checkpoint["best_score"]

    loss_fn = LossRate_H().to(device)

    train_dataset = TripletDataset(train_data_str)
    eval_dataset = TripletDataset(dev_data_str)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size, shuffle=False) # no need to shuffle

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)
    for epoch in range(epoch_range):
        model.train()
        labels = []
        model.normalizor()
        for batch in tqdm.tqdm(train_loader):
            h = batch["h"].to(device)
            r = batch["r"].to(device)
            t = batch["t"].to(device)
            # optimizing model
            score, neg_score, scale_loss, orthogonal_loss = model((h, r, t))
            optimizer.zero_grad()
            loss = loss_fn(score, neg_score, scale_loss, orthogonal_loss)
            loss.backward()
            optimizer.step()
            model.normalizor()
            labels.append(batch["t"].flatten().cpu())

        model.eval()
        hit_1_num = 0
        hit_5_num = 0
        hit_10_num = 0
        develop_num = 0
        for eval_batch in tqdm.tqdm(eval_loader):
            h = eval_batch["h"].to(device)
            r = eval_batch["r"].to(device)
            predicton_results = model.predict((h, r))
            true_labels = eval_batch["t"].flatten().cpu().tolist()
            develop_num += len(true_labels)
            for index in range(len(true_labels)):
                hit_1 = predicton_results["hit@1"][index]
                hit_5 = predicton_results["hit@5"][index]
                hit_10 = predicton_results["hit@10"][index]
                if true_labels[index] in hit_1:
                    hit_1_num += 1
                if true_labels[index] in hit_5:
                    hit_5_num += 1
                if true_labels[index] in hit_10:
                    hit_10_num += 1
        hit_1_acc = hit_1_num / develop_num
        hit_5_acc = hit_5_num / develop_num
        hit_10_acc = hit_10_num / develop_num
        curr_score = 0.3 * hit_1_acc + 0.7 * hit_5_acc   # hit@1 most
        if curr_score > best_score:
            best_score = curr_score
            torch.save({"entity_word2index": entity_pre_embedded["word2index"],
                        "relation_word2index": relation_pre_embedded["rel2idx"],
                        "model_state_dict": model.state_dict(),
                        "hit@1": hit_1_acc,
                        "hit@5": hit_5_acc,
                        "best_score": best_score}, train_model_save_str)
        print("epoch = ", epoch + 1, "Hit@1 = ", hit_1_acc, "Hit@5 = ", hit_5_acc, "Hit@10 = ", hit_10_acc, "\n")
train()
