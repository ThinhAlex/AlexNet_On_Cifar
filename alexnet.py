### I. Import libraries 
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import random
import matplotlib.pyplot as plt
random.seed(0)

### II. Utility functions
# a. create model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=1024, out_features=10),
        )
                 
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# b. train
def train(epochs, model, train_loader, criterion, optimizer, device):
    train_loss_list = []
    train_acc_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        sample = 0
        for idx, (X_train, y_train) in enumerate(train_loader):
            # to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # forward
            optimizer.zero_grad()
            pred = model(X_train)    
            
            # calculate correct predictions
            correct += (torch.argmax(pred, dim = 1) == y_train).sum().item()
            sample += y_train.size(0)
            
            # loss
            loss = criterion(pred, y_train)
            total_loss += loss.item() 
            
            # backward
            loss.backward()
            
            # step
            optimizer.step()
        
        train_acc = (correct/sample)*100
        train_acc_list.append(train_acc)
        avg_loss = total_loss/(len(train_loader))
        train_loss_list.append(avg_loss)
        
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {avg_loss:.2f} | Accuracy: {train_acc:.2f}%")
    
    return train_acc_list, train_loss_list, model

def show_loss(result_list, epoch):
    epoch_list = torch.range(1, epoch, 1)   
    plt.plot(epoch_list, result_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend("Train Loss")
    plt.show()
    
def show_acc(result_list, epoch):
    epoch_list = torch.range(1, epoch, 1)   
    plt.plot(epoch_list, result_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend("Train Accuracy")
    plt.show()

# c. evaluate on test data
def evaluate(model, test_loader, num_class, device):
    with torch.no_grad():
        model.eval()
        correct = 0
        sample = 0
        
        class_correct = [0 for i in range(num_class)]
        class_sample = [0 for i in range(num_class)]
        for X_test, labels in test_loader:
            X_test = X_test.to(device)
            labels = labels.to(device)
            
            pred = model(X_test)
            pred_classes = torch.argmax(pred, dim = 1)
            correct += (pred_classes == labels).sum().item()
            sample += labels.size(0)
            
            for i in range(len(pred_classes)):
                label = labels[i]
                pred_class = pred_classes[i]
                
                if label == pred_class:
                    class_correct[label] += 1
                class_sample[label] += 1
        test_acc = correct/sample*100
        
        print(f"Model accuracy: {test_acc}%")
        
        for i in range(num_class):
            class_acc = class_correct[i]/class_sample[i]*100
            print(f"Accuracy of class {i}: {class_acc:.2f}%") 
        
def save_model(model, PATH):
    torch.save(model, PATH)

def load_model(PATH):
    model = torch.load(PATH)
    return model

if __name__ == "__main__":   
    ### III. Recommended setup code
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # Setup dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,0.5,0.5), (0.2,0.2,0.2)),
    ]) 
    train_data = datasets.CIFAR10(root = "dataset/", train=True, transform=transform, download = True)
    test_data = datasets.CIFAR10(root = "dataset/", train=False, transform=transform, download = True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

    # Initialize model, loss, optimizer
    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    train_acc_list, train_loss_list, model = train(epochs, model, train_loader, criterion, optimizer, device)

    # Save model
    save_model(model, "ckpt/alexnet_cifar10.pth")
    
    # Evaluate model
    evaluate(model, test_loader, num_class=10, device=device)   
    
    # Show loss and accuracy
    show_loss(train_loss_list, epochs)
    show_acc(train_acc_list, epochs)
    

