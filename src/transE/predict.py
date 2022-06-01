import pickle 
from transE import TransE
import torch
from tripletGens import TestTripletDataset
import tqdm

device = "cuda:0"
entity_pre_emb = pickle.load(open("./output/entity_embedded.bin", "rb"))
relation_pre_emb = pickle.load(open("./output/relation_embedded.pkl", "rb"))

model = TransE(entity_pre_emb["word2index"],
                relation_pre_emb["rel2idx"],
                embed_dimension = 100)

checkpoint = torch.load("./output/transEtrained.bin", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

test_dataset = TestTripletDataset("./data/test.txt")

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle=False)

with open("./submit/result.txt", "w", encoding="utf-8") as fw:
    for test_batch in tqdm.tqdm(dataloader):
        h = test_batch["h"].to(device)
        r = test_batch["r"].to(device)
        prediction = model.predict((h, r))
        for result in prediction["hit@5"]:
            str_result = ''
            for item in result:
                str_result += str(item) + ","
            str_result = str_result[:-1]
            fw.write(str_result + '\n')