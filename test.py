from utils import get_arguments, load_checkpoint
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.6583, 0.4580, 0.0877], [0.2412, 0.2313, 0.2387]
train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])

def test_video(model, video):
    cap = cv2.VideoCapture(video)

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
                
                result = model(image)
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

def test_image(model, image_name):
    image = Image.open(image_name)
    tensor = train_transforms(image).unsqueeze(0)
    
    images = tensor.to(device)
    outputs = model(images)
    _, pred = torch.max(outputs.data, 1)
    return pred.item()

args = get_arguments()

if args.video:
    loaded_model = load_checkpoint(args.model_path)
    if args.video == "0":
        test_video(loaded_model, 0)
    else:
        test_video(loaded_model, args.video)
elif args.image:
    loaded_model = load_checkpoint(args.model_path)
    test_image(loaded_model, args.image)