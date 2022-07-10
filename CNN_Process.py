import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import cv2
import pandas as pd

# ĐỊNH DẠNH ẢNH ĐẦU VÀO
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

#TẠO MODEL
class Cnn(nn.Module):
  def __init__(self):
    super(Cnn,self).__init__()
    self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
    self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    self.fc1 = nn.Linear(3*3*64,10)
    self.fc2 = nn.Linear(10,2)
    self.relu = nn.ReLU()
        
  def forward(self,x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = out.view(out.size(0),-1)
      out = self.relu(self.fc1(out))
      out = self.fc2(out)
      return out


def load_model(model,save_path):
    load_weights = torch.load(save_path)
    model.load_state_dict(load_weights)
    return model

class_index = ["free", "busy"]

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        predict_label = self.clas_index[max_id]
        return predict_label


predictor = Predictor(class_index)

resize = (224,224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
save_path = 'model.pth'
model = Cnn()

def predict(img):
    model = Cnn()
    model.eval()
    model = load_model(model, save_path)
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)
    output = model(img)
    response = predictor.predict_max(output)
    return response

def Process_Image(img,img_show):
    df = pd.read_csv(r"CNN_slot.csv")
    index_list_busy = []
    index_list = []
    for index in df['index']:
        x = df['x'][index - 1] + 35
        y = df['y'][index - 1] + 20
        w = df['w'][index - 1] + x
        h = df['h'][index - 1] + y
        area = (x, y, w, h)
        cv2.putText(img_show,f"{index}",(x-5,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        cropped_img = img.crop(area)
        predicted = predict(cropped_img)

        if predicted == "busy":
            index_list_busy.append(index)
            index_list.append(index)
            cv2.rectangle(img_show, (x, y), (w, h), (0, 0, 255), 3)
        else:
            index_list.append(index)
            cv2.rectangle(img_show, (x, y), (w, h), (0, 255, 0), 3)

    img_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb,(1280,720))

    f = open("Status_CNN.csv","w+")
    f.write("slot,status\n")
    for i in index_list:
        if i in index_list_busy:
            a = "Busy"
        else:
            a = "Free"
        f.write(f"{ i },{ a }\n")
    f.close()
    
    return img
