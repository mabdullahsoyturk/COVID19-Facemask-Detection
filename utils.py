import os
import glob
import xmltodict
import cv2
import torch
import matplotlib.pyplot as plt
import argparse

classes = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
IMAGE_DIR = 'dataset/images/'
ANNOTATION_DIR = 'dataset/annotations/'
TRAIN_DIR = 'train/'

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Mask Classifier Arguments')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-d', '--weight-decay', default=1e-5, type=float)
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--model-path', default="models/9.pth")
    parser.add_argument('-a', '--arch', default='resnet18')

    return parser.parse_args()

def get_image_names():
    paths = glob.glob("dataset/images/*")
    image_names = [path.rsplit("/",1)[1] for path in paths]
    
    return image_names

def get_paths(image_name):
    image_path = IMAGE_DIR + image_name
    xml_name = image_name[:-4] + '.xml'
    annotation_path = ANNOTATION_DIR + xml_name
        
    return  image_path, annotation_path

def parse_xml(annotation_path):
    xml = xmltodict.parse(open(annotation_path , 'rb'))
    item_list = xml['annotation']['object']
    
    # when image has only one bounding box
    if not isinstance(item_list, list):
        item_list = [item_list]
        
    result = []
    
    for item in item_list:
        name = item['name']
        bndbox = [(int(item['bndbox']['xmin']), int(item['bndbox']['ymin'])),
                  (int(item['bndbox']['xmax']), int(item['bndbox']['ymax']))]       
        result.append((name, bndbox))
    
    return result
    
def create_directory(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
        # print("Directory " + dirname + " already exists.")

def create_directories():
    directories = [TRAIIN_DIR + "0/", TRAIN_DIR + "1/", TRAIN_DIR + "2/", "models/"]

    for directory in directories:
        create_directory(directory)

def crop_image(image_name):
    image_path, annotation_path = get_paths(image_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    labels = parse_xml(annotation_path)
    
    cropped_pairs = []
    for name, bndbox in labels:
        cropped_image = image[bndbox[0][1]:bndbox[1][1], bndbox[0][0]:bndbox[1][0]]
        label = classes[name]
        
        croppped_pair = [cropped_image, label]
        cropped_pairs.append(cropped_pair)
        
    return cropped_pairs

def prepare_dataset():
    image_names = get_image_names()
    
    for image_name in image_names:
        cropped_pairs = crop_image(image_name)
        
        for index, img, label in enumerate(cropped_pairs):
            cropped_img_name = str(index) + ".jpg"
            cv2.imwrite(TRAIN_DIR + str(label) + "/" + cropped_img_name, img)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    return model.eval()

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0,2,3])
        num_batches +=1

    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean ** 2) ** 0.5

    return mean,std

def save_loss_fig(NUM_EPOCH, training_losses, val_losses):
    plt.figure(1)
    plt.plot(range(NUM_EPOCH), training_losses, label='Training')
    plt.plot(range(NUM_EPOCH), val_losses, label='Validation')
    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses")

def save_accuracy_fig(NUM_EPOCH, training_accuracies, val_accuracies):
    plt.figure(2)
    plt.plot(range(NUM_EPOCH), training_accuracies, label='Training')
    plt.plot(range(NUM_EPOCH), val_accuracies, label='Validation')
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracies")
