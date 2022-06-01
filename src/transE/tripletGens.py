import torch
import torch.utils.data

# build our own dataset via data provided

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, datafile: str) -> None:
        super(TripletDataset, self).__init__()
        self.triplets = {"h": [], "r": [], "t": []}
        self.triplets_sum = 0
        with open(datafile, "r", encoding = "utf-8") as fr:
            for line in fr:
                h, r, t = line.strip().split("\t")
                self.triplets["h"].append(int(h))
                self.triplets["r"].append(int(r))
                self.triplets["t"].append(int(t))
                self.triplets_sum = self.triplets_sum + 1
        self.triplets["h"] = torch.LongTensor(self.triplets["h"])
        self.triplets["r"] = torch.LongTensor(self.triplets["r"])
        self.triplets["t"] = torch.LongTensor(self.triplets["t"])

    def __getitem__(self, idx):
        triplets = {key: torch.LongTensor([val[idx]]) for key, val in self.triplets.items()}
        return triplets

    def __len__(self):
        return self.triplets_sum


class TestTripletDataset(torch.utils.data.Dataset):
    def __init__(self, datafile: str) -> None:
        super(TestTripletDataset, self).__init__()
        self.testTriplets = {"h": [], "r": []}
        self.testTriplets_sum = 0
        with open(datafile, "r", encoding = "utf-8") as fr:
            for line in fr:
                h, r, t = line.strip().split("\t")  # t is ?
                self.testTriplets["h"].append(int(h))
                self.testTriplets["r"].append(int(r))
                self.testTriplets_sum = self.testTriplets_sum + 1
        self.testTriplets["h"] = torch.LongTensor(self.testTriplets["h"])
        self.testTriplets["r"] = torch.LongTensor(self.testTriplets["r"])

    def __getitem__(self, idx):
        testTriplets = {key: torch.LongTensor([val[idx]]) for key, val in self.testTriplets.items()}
        return testTriplets

    def __len__(self):
        return self.testTriplets_sum
