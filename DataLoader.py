import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


class CustomDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        self.x = torch.tensor(self.x, dtype = torch.float)
        self.y = torch.tensor(self.y, dtype = torch.float)
        return self.x[idx],self.y[idx]

# %%

def prepareDataLoader(DNAListX, DNAListY):
    DS = CustomDataSet(DNAListX, DNAListY)
    tlen = int(0.8*len(DS))
    tstlen = len(DS) - tlen
    trainDS, testDS = torch.utils.data.random_split(DS, [tlen, tstlen])
    trainDL = DataLoader(trainDS, batch_size = 10, shuffle = True)
    testDL = DataLoader(testDS, batch_size = 10, shuffle = True)
    print(tlen, tstlen())
    return trainDL, testDL

        
