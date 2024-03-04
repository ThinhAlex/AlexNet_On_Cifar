from alexnet_pretrained_cifar10 import *

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_class = 100
batch_size = 64
learning_rate = 0.001
epochs = 15

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227,227), antialias=True),
    transforms.CenterCrop((224,224)),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
]) 
train_data = datasets.CIFAR100(root = "dataset/", train=True, transform=transform, download = True)
test_data = datasets.CIFAR100(root = "dataset/", train=False, transform=transform, download = True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

# fine tuning model, loss, optimizer
model = torch.load("pretrained_cifar10/alexnet_pretrained_cifar10.pth")
for param in model.parameters():
    param.requires_grad = False
    
add_layers = nn.Sequential(
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 100),
)   
model.classifier[5] = nn.Linear(4096, 4096)
model.classifier[7] = nn.Linear(4096, 1024)

model.classifier = nn.Sequential(
    model.classifier,
    add_layers,
)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

# train
train_acc_list, train_loss_list, model = train(epochs, model, train_loader, criterion, optimizer, device)

# save model
save_model(model, "transfer_cifar100/alexnet_transfer_cifar100.pth")
print("=> Model saved!")

# show train loss
show_loss(train_loss_list, epochs)
show_acc(train_acc_list, epochs)

# evaluate test data
evaluate(model, test_loader, num_class, device)



