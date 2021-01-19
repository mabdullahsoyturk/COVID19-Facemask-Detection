import numpy as np
import time
import os
import cv2
import glob
from PIL import Image
import torchvision 
import torch 
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from utils import load_checkpoint, save_loss_fig, save_accuracy_fig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.6583, 0.4580, 0.0877], [0.2412, 0.2313, 0.2387]
BATCH_SIZE, LEARNING_RATE, NUM_EPOCH, WEIGHT_DECAY = 16, 1e-4, 20, 1e-5

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
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
# dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 1000, replace=False))

dataset_size = len(dataset)
train_size, val_size = int(dataset_size * 0.6),  int(dataset_size * 0.2)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f'Dataset size: {dataset_size}\nTrain set size: {len(train_dataset)}\nValidation set size: {len(val_dataset)}\nTest set size: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

best_epoch = 0

def train(model):
    training_losses, val_losses, training_accuracies, validation_accuracies = [], [], [], []

    for epoch in range(NUM_EPOCH):
        epoch_train_loss, correct, total = 0,0,0

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
            total += y.size(0)
            correct += (maximum == y).sum().item()
        
        training_accuracy = correct / total
        training_losses.append(epoch_train_loss / total)
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
        print(f'EPOCH:{epoch}, Training Loss:{epoch_train_loss / total}, Validation Loss:{epoch_val_loss / total}, Training Accuracy: {training_accuracy}, Validation Accuracy: {accuracy}')
        
        if min(val_losses) == val_losses[-1]:
            best_epoch = epoch
            checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint, "models/" + f'{epoch}.pth')
            print("Model saved")

        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    save_loss_fig(NUM_EPOCH, training_losses, val_losses)
    save_accuracy_fig(NUM_EPOCH, training_accuracies, validation_accuracies)

start_time = time.time()
#train(model)
end_time = time.time()
duration = end_time - start_time
print(f'Time it takes to train {duration}')

#filepath = f'models/{best_epoch}.pth'
filepath = f'models/{9}.pth'
loaded_model = load_checkpoint(filepath)

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

test(loaded_model, test_loader)

def test_video(model):
    cap = cv2.VideoCapture(0)

    font_scale, thickness = 1, 2
    red,green,blue = (0,0,255), (0,255,0), (255,0,0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rectangle_spec = {0: ("Masked", green), 1: ("No Mask", red), 2: ("Incorrect Mask", blue)}

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.4, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), blue, 2)
                
                cropped_img = frame[y:y+h, x:x+w]
                pil_image = Image.fromarray(cropped_img, mode = "RGB")
                pil_image = train_transforms(pil_image)
                image = pil_image.unsqueeze(0)
                
                result = loaded_model(image)
                _, maximum = torch.max(result.data, 1)
                prediction = maximum.item()
                print(prediction)
                
                text, color = rectangle_spec[prediction]
                cv2.putText(frame, text, (x,y - 10), font, font_scale, color, thickness)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            cv2.imshow('frame',frame)
            
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def transform_image():
    image = Image.open("diff_mask.jpeg")

    return train_transforms(image).unsqueeze(0)

def get_prediction(image_tensor):
    images = image_tensor.to(device)
    outputs = loaded_model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

tensor = transform_image()
prediction = get_prediction(tensor)
print(prediction.item())

# test_video(loaded_model)
