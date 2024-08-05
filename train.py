import time
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import random


def get_melspec(file_path, nfft_, ht_, sr=8000, top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=512, n_fft=nfft_, hop_length=ht_)
    spec = librosa.power_to_db(spec, top_db=top_db)
    
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    
    spec_min, spec_max = spec.min(), spec.max()
    spec = 255 * (spec - spec_min) / (spec_max - spec_min)
    spec = spec.astype(np.uint8)
    spec = spec[np.newaxis, ...]
    return spec


class ListDataset(Dataset):
    def __init__(self, label_file, label_list, nfft_, ht_):
        self.label_file = pd.read_csv(label_file)
        self.label_list = label_list
        self.transform = transforms.ToTensor()

        self.specs = []
        self.labels = []

        for i in range(len(self.label_file)):
            audio_path = self.label_file.iloc[i]['Fname']
            spec = get_melspec(audio_path, nfft_, ht_)

            self.specs.append(spec)

            label_class = self.label_file.iloc[i]['Species']
            label = torch.tensor(self.label_list[label_class])
            self.labels.append(label)

    def __getitem__(self, index):
        cur_spec = self.specs[index]
        return cur_spec, self.labels[index]
    
    def __len__(self):
        #         length of the whole dataset
        return len(self.labels)
    

class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet121, self).__init__()
        # Load pre-trained DenseNet121 model
        self.densenet121 = models.densenet121(pretrained=True)
        
        # Modify the input layer to accept single-channel images
        self.densenet121.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the output layer to have the desired number of classes
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet121(x)

def denseNet121(n_classes):
    return CustomDenseNet121(n_classes)
    

def train(model, device, train_loader, valid_loader, epochs, optimizer, sfname):
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 0)
    lf=nn.CrossEntropyLoss()
    best_valid_acc = 0
    best_model_report = ''

    # training procedure
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        loss_val = 0
        true_running = 0
        total_running = 0
        for i, data in enumerate(train_loader):
            x, gt = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            predicted = model(x)
            loss = lf(predicted, gt)

            result, predicted_class = torch.max(predicted, 1)
            true_running += (predicted_class == gt).sum()
            total_running += predicted_class.shape[0]

            loss.backward()
            optimizer.step()

            loss_val += loss.item()

        train_loss = loss_val / len(train_loader)
        accuracy = torch.true_divide(true_running, total_running)

        sched.step()
        model.eval()

        # validating procedure
        valid_loss_val = 0
        valid_true_running = 0
        valid_total_running = 0
        y_pred = np.array([])
        y_test = np.array([])
        for i, data in enumerate(valid_loader):
            x, gt = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)
            predicted = model(x)
            loss = lf(predicted, gt)

            result, predicted_class = torch.max(predicted, 1)
            valid_true_running += (predicted_class == gt).sum()
            valid_total_running += predicted_class.shape[0]

            valid_loss_val += loss.item()

            y_pred = np.append(y_pred, predicted_class.cpu().detach().numpy())
            y_test = np.append(y_test, gt.cpu().detach().numpy())

        # calculating measurements
        valid_loss = valid_loss_val / len(train_loader)
        accuracy = torch.true_divide(valid_true_running, valid_total_running)

        # time usage for each epoch
        end_time = time.time()
        usage_time = end_time - start_time

        # save best model and its performance report, can be used for futher training
        if accuracy > best_valid_acc:
            best_valid_acc = accuracy
            best_model_report = classification_report(y_test, y_pred, zero_division=0)
            torch.save(model.state_dict(), sfname)
        if epoch == epochs:
            with open('log.txt', 'a+') as file:
                file.write(f'Best Validation Accuracy: {best_valid_acc} \n')
            

if __name__ == "__main__":
    with open('log.txt', 'w') as file:
        file.write(f'Ready! \n')
        file.write(f'Mel bands: 512 \n\n')
    train_csv = "/home/seshasaianeeshteja_vempa_student_uml_edu/Stanford_trainData_Abuzz.csv"
    valid_csv = "/home/seshasaianeeshteja_vempa_student_uml_edu/Stanford_valiData_Abuzz.csv"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 25
    
    label_list = {'aegypti': 0, 'albopictus': 1, 'mediovittatus': 2, 'sierrensis': 3, 'albimanus': 4, 'arabiensis': 5, 
                  'atroparvus': 6, 'dirus': 7, 'farauti': 8, 'freeborni': 9, 'gambiae': 10, 'merus': 11, 'minimus': 12, 
                  'quadriannulatus': 13, 'quadrimaculatus': 14, 'stephensi': 15, 'pipiens': 16, 'quinquefasciatus': 17, 
                  'tarsalis': 18, 'incidens': 19}
    
    nfft = [1024]
    ht = [64,96,108]
    lr = [2e-4,2e-5,2e-6]
    
    for nfft_ in nfft:
        for ht_ in ht:
            sfname = './stanford20_512_'+str(nfft_)+'_'+str(ht_)+'__0'+'.pth'
            with open('log.txt', 'a+') as file:
                file.write(f'\nwindow size: {nfft_},\nhop length: {ht_} \n')
            for lr_ in lr:
                train_data = ListDataset(train_csv, label_list, nfft_, ht_)
                vali_data = ListDataset(valid_csv, label_list, nfft_, ht_)

                train_loader = DataLoader(train_data, batch_size, shuffle=True)
                vali_loader = DataLoader(vali_data, batch_size, shuffle=True)

                path_to_checkpoint = sfname
                resnet_model = denseNet121(20)
                
                if lr_!=2e-4:
                    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    resnet_model.load_state_dict(checkpoint)
                    sfname = sfname.split('__')[0]+'__'+str(int(sfname.split('__')[1][0])+1)+sfname.split('__')[1][1:]
                    print('sfname',sfname)

                resnet_model = resnet_model.to(device)
                optimizer = optim.Adam(resnet_model.parameters(), lr=lr_)

                train(resnet_model, device, train_loader, vali_loader, epochs, optimizer, sfname)
