import torch
import torchvision
from dataprocessing import UniqueNoisyDNASequences
from DataLoader import prepareDataLoader
from model import Autoencoder
from train import trainf, testf
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device



if __name__=='__main__':
    epoch = 100
    lra = 1e-3
    path = '../../../data/DeepIntegrate_Data_HOH_2019.txt'
    device = get_device()
    
    DNAListX, DNAListY = UniqueNoisyDNASequences(path)
    trainDL, testDL = prepareDataLoader(DNAListX, DNAListY)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lra)
    net = Autoencoder()
    
    train_loss = train(device, net, trainDL, epoch, criterion, optimizer)
    test_loss = testf(device, net, testDL, epoch, criterion)

#     plt.plot(train_loss)
#     plt.title('Train Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
