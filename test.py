import os
import cv2
import glob
import matplotlib.pyplot as plt
import random
from PIL import Image
import torchvision 
import torch 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import get_path, parse_xml, create_directories, load_checkpoint, save_loss_fig, save_accuracy_fig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
mean = [0.6583, 0.4580, 0.0877]
# mean = [0.485, 0.456, 0.406]
std = [0.2412, 0.2313, 0.2387]
# std = [0.229, 0.224, 0.225] 
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCH = 30
WEIGHT_DECAY = 0

# create_directories()

model = models.mobilenet_v2()
# model = models.resnet50()
# model = models.vgg19()
# model = models.densenet161()
# model = models.resnet34(pretrained=True)
#for layer, param in model.named_parameters():
#    if 'fc' not in layer:
#        param.requires_grad = False

#n_inputs = model.fc.in_features
#last_layer = torch.nn.Linear(n_inputs, 3)
#model.fc.out_features = last_layer
model.classifier._modules['1'] = torch.nn.Linear(1280, 3)
# model.classifier = torch.nn.Linear(2208,3)
if torch.cuda.device_count() > 1:
  print("Let's use" + str(torch.cuda.device_count()) + "GPUs!")
  model = torch.nn.DataParallel(model)
train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])

dataset = datasets.ImageFolder("train", transform = train_transforms)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.6)
val_size = int(dataset_size * 0.2)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print(f'Dataset size: {dataset_size}\nTrain set size: {len(train_dataset)}\nValidation set size: {len(val_dataset)}\nTest set size: {len(test_dataset)}')

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.to(device)

best_epoch = 0
training_losses, val_losses, training_accuracies, validation_accuracies = [], [], [], []

for epoch in range(NUM_EPOCH):
    epoch_train_loss = 0
    correct = 0
    total = 0

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
    training_losses.append(epoch_train_loss)
    training_accuracies.append(training_accuracy)

    epoch_val_loss = 0
    correct = 0
    total = 0
    
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
            
    val_losses.append(epoch_val_loss)
    accuracy = correct/total
    validation_accuracies.append(accuracy)
    print(f'EPOCH:{epoch}, Training Loss:{epoch_train_loss}, Validation Loss:{epoch_val_loss}, Training Accuracy: {training_accuracy}, Validation Accuracy: {accuracy}')
    
    if min(val_losses) == val_losses[-1]:
        best_epoch = epoch
        checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}

        torch.save(checkpoint, "models/" + f'{epoch}.pth')
        print("Model saved")

save_loss_fig(NUM_EPOCH, training_losses, val_losses)
save_accuracy_fig(NUM_EPOCH, training_accuracies, validation_accuracies)
# plt.show()

filepath = f'models/{best_epoch}.pth'
loaded_model = load_checkpoint(filepath)

correct = 0
total = 0
    
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        result = loaded_model(X)
        _, maximum = torch.max(result.data, 1)
        print(maximum)
        total += y.size(0)
        correct += (maximum == y).sum().item()

accuracy = correct/total
print(f'Testing Accuracy: {accuracy}')

""" cap = cv2.VideoCapture(0)

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
            
            croped_img = frame[y:y+h, x:x+w]
            pil_image = Image.fromarray(croped_img, mode = "RGB")
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
cv2.destroyAllWindows() """
