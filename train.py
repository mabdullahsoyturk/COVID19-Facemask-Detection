import numpy as np
import time
from PIL import Image
import torchvision 
import torch 
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from utils import prepare_dataset, load_checkpoint, save_loss_fig, save_accuracy_fig, get_arguments

args = get_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.6583, 0.4580, 0.0877], [0.2412, 0.2313, 0.2387]
BATCH_SIZE, LEARNING_RATE, NUM_EPOCH, WEIGHT_DECAY = args.batch_size, args.learning_rate, args.epochs, args.weight_decay

models = {
    'resnet18': torchvision.models.resnet18(),
    'resnet34': torchvision.models.resnet34(),
    'resnet50': torchvision.models.resnet50(),
    'vgg16': torchvision.models.vgg16(),
    'vgg19': torchvision.models.vgg19(),
    'densenet': torchvision.models.densenet161(),
    'mobilenet': torchvision.models.mobilenet_v2()
}

def get_model(model_name):
    model = models[model_name]

    if "resnet" in model_name:
        n_inputs = model.fc.in_features
        last_layer = torch.nn.Linear(n_inputs, 3)
        model.fc.out_features = last_layer
    elif model_name == 'mobilenet':
        model.classifier._modules['1'] = torch.nn.Linear(1280, 3)
    elif "vgg" in model_name:
        model.classifier._modules['6'] = torch.nn.Linear(4096, 3)
    elif model_name == 'densenet':
        model.classifier = torch.nn.Linear(2208,3)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    return model

model = get_model("resnet18")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

# prepare_dataset()

train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
augment_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(p=1),
    torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])

dataset = datasets.ImageFolder("train", transform = train_transforms)
augmented_dataset = datasets.ImageFolder("train", transform = augment_transforms)
dataset = torch.utils.data.ConcatDataset([augmented_dataset,dataset])

dataset_size = len(dataset)
train_size, val_size = int(dataset_size * 0.6),  int(dataset_size * 0.2)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f'Dataset size: {dataset_size}\nTrain set size: {len(train_dataset)}\nValidation set size: {len(val_dataset)}\nTest set size: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

best_model = None

def train(model):
    training_losses, val_losses, training_accuracies, validation_accuracies = [], [], [], []

    for epoch in range(NUM_EPOCH):
        epoch_train_loss, correct, train_total = 0,0,0

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            result = model(X)
            loss = criterion(result, y)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, maximum = torch.max(result.data, 1)
            train_total += y.size(0)
            correct += (maximum == y).sum().item()
        
        training_accuracy = correct / train_total
        training_losses.append(epoch_train_loss / train_total)
        training_accuracies.append(training_accuracy)

        epoch_val_loss, correct, total = 0,0,0
        
        with torch.no_grad():
            model.eval()
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                 
                result = model(X)
                loss = criterion(result, y)
                epoch_val_loss += loss.item()
                _, maximum = torch.max(result.data, 1)
                total += y.size(0)
                correct += (maximum == y).sum().item()
                
        val_losses.append(epoch_val_loss / total)
        accuracy = correct/total
        validation_accuracies.append(accuracy)
        print(f'EPOCH:{epoch}, Training Loss:{epoch_train_loss / train_total}, Validation Loss:{epoch_val_loss / total}, Training Accuracy: {training_accuracy}, Validation Accuracy: {accuracy}')
        
        if min(val_losses) == val_losses[-1]:
            checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
            best_model = "models/" + f'{epoch}.pth'
            torch.save(checkpoint, best_model)
            print("Model saved")

        scheduler.step(epoch_val_loss)
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    save_loss_fig(NUM_EPOCH, training_losses, val_losses)
    save_accuracy_fig(NUM_EPOCH, training_accuracies, validation_accuracies)

start_time = time.time()
train(model)
end_time = time.time()
duration = end_time - start_time
print(f'Time it takes to train {duration}')

def test(model, test_loader):
    correct, total = 0, 0
        
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            result = model(X)
            _, maximum = torch.max(result.data, 1)
            total += y.size(0)
            correct += (maximum == y).sum().item()

    accuracy = correct/total
    print(f'Testing Accuracy: {accuracy}')

loaded_model = load_checkpoint(best_model)
test(loaded_model, test_loader)