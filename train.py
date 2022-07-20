import torch
import torchvision
import torch.nn as nn
import numpy as np

def cal_accuracy(outputs, target):
    score = 0
    outputs = np.array(outputs.detach().cpu().numpy())
    target = np.array(target.detach().cpu().numpy())
    target = target.transpose(0,2,1)    #batch_size * 13714 * 8
    
    outputs = outputs.transpose(0,2,1)   #batch_size * 13714 * 8
    out = np.zeros(outputs.shape)
    idx = outputs.argmax(axis=-1)
    out[np.arange(outputs.shape[0])[:,None],np.arange(outputs.shape[1]),idx] = 1  # convert output in one hot encoding format by making only highest value to 1 and 0 to rest
    
    score = np.sum(np.all((out == target), axis = 2))/(outputs.shape[0]*outputs.shape[1])
#     print(score)
    return score

def trainf(device, net, trainloader, epoch, criterion, optimizer):
    train_loss = []
    for epoch in range(epoch):
        running_loss = 0.0
        correct = 0
        res = 0
        net = net.to(device)
        
        for data in trainloader:
            img, y = data 
#             img = img.view(10, 1, 123426)
            img = img.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, y)
            res += cal_accuracy(outputs,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
     
        loss = running_loss / len(trainloader)
        accuracy = (res*100) / len(trainloader)
        train_loss.append(loss)
        print('Epoch {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}'.format(
            epoch+1, loss, accuracy))
    PATH ='model.pth'
    torch.save(net.state_dict(), PATH)    
    return train_loss

def testf(device, net, testloader, epoch, criterion):
    test_loss = []
    
    running_loss = 0.0
    res = 0
    
    loadModelFile = 'model.pth'
    net.load_state_dict(torch.load(loadModelFile))
    net.eval()
    net = net.to(device)
    
    for data in testloader:
      img, y = data 
      img = img.to(device)
      y = y.to(device)
      outputs = net(img)

      loss = criterion(outputs, y)
      running_loss += loss.item()
      res += cal_accuracy(outputs,y)
        
    loss = running_loss / len(testloader)
    accuracy = (res*100) / len(testloader)
    test_loss.append(loss)
    print('Epoch {} of {}, Test Loss: {:.3f}, Test Accuracy: {:.3f}'.format(epoch+1, epoch, loss, accuracy))
    return test_loss
