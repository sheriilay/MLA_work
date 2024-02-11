#main libs
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as opt
#visual libs
import matplotlib.pyplot as plt
import numpy as np
#defining paths, transforms and creating dataset from images
mean = [0.4725, 0.4614, 0.4750]
std = [0.1818, 0.2227, 0.2390]

trainset_path = 'dataset/train'
testset_path = 'dataset/test'

trainset_transforms =  transforms.Compose([transforms.Resize((200,200)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])
testset_transforms = transforms.Compose([transforms.Resize((200,200)),transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])

trainset = torchvision.datasets.ImageFolder(root=trainset_path, transform=trainset_transforms)
testset = torchvision.datasets.ImageFolder(root = testset_path, transform=testset_transforms)

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size=32, shuffle=False)

def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print('Epoch number %d' %(epoch+1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        print("-Training dataset. Got %d out o %d images correctly (%.3f%%). Epoch loss: %.3f" % (running_correct, total, epoch_acc, epoch_loss))
        test_data_acc = eval_model_on_test(model, test_loader)  
        if(test_data_acc>best_acc):
            best_acc = test_data_acc
            save_checkpoint(model,epoch,optimizer,best_acc)
          
    print("Finished")
    return model
    
def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch+1,
        'model': model.state_dict(),
        'best accuracy' : best_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')

def eval_model_on_test(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_correctly_on_epoch += (predicted==labels).sum().item()
    
    epoch_acc = 100.0 * predicted_correctly_on_epoch/total
    print(" -Testing dataset. Got %d out of %d images correctly(%.3f%%)"%(predicted_correctly_on_epoch,total,epoch_acc))
    return epoch_acc

resnet18_model = models.resnet18(pretrained=True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 28
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = set_device()
resnet18_model =  resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = opt.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 5)

checkpoint = torch.load('model_best_checkpoint.pth.tar')

resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 28
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model, 'model.pth')

"""
#func to show images
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    
    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    print('labels: ', labels)
    plt.imshow(np.transpose(grid,(1,2,0)))
    plt.show()

show_transformed_images(trainset)

#calculating mean and standart deviation to proper transform
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_img_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_img_count += image_count_in_a_batch
    mean /= total_img_count
    std /= total_img_count
    
    return mean, std

print(get_mean_and_std(train_loader))
"""