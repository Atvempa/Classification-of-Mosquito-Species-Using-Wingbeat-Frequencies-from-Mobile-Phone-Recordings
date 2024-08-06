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
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotConfusionMatrix(a):
    row_sums = a.sum(axis=1)

    # Adjusting labels for plotting
    labels = (np.asarray(["{0:}\n----\n{1:}".format(value, row_sums[i])
                for i, row in enumerate(a)
                for value in row])
              ).reshape(20,20)

    # Labels for axis
    labels_list = ['Ae aegypti', 'Ae albopictus', 'Ae mediovittatus', 'Ae sierrensis', 'An albimanus', 
                   'An arabiensis', 'An atroparvus', 'An dirus', 'An farauti', 'An freeborni', 'An gambiae', 
                   'An merus', 'An minimus', 'An quadriannulatus', 'An quadrimaculatus', 'An stephensi', 
                   'C pipiens', 'C quinquefasciatus', 'C tarsalis', 'Cu incidens']

    # Plotting the confusion matrix
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=0.85)
    ax = sns.heatmap(a, annot=labels, fmt='', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Testing Data Confusion Matrix')
    plt.show()
    return

    
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



if __name__ == "__main__":
    
    path_to_checkpoint = '/home/seshasaianeeshteja_vempa_student_uml_edu/ondemand/data/sys/myjobs/projects/default/3/dnet121f_1024_108__2.pth'

    # Load the model checkpoint
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cuda'))

    model = denseNet121(20)
    model.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    
    label_list = {'aegypti': 0, 'albopictus': 1, 'mediovittatus': 2, 'sierrensis': 3, 'albimanus': 4, 'arabiensis': 5, 
                  'atroparvus': 6, 'dirus': 7, 'farauti': 8, 'freeborni': 9, 'gambiae': 10, 'merus': 11, 'minimus': 12, 
                  'quadriannulatus': 13, 'quadrimaculatus': 14, 'stephensi': 15, 'pipiens': 16, 'quinquefasciatus': 17, 
                  'tarsalis': 18, 'incidens': 19}

    v_csv = "/home/seshasaianeeshteja_vempa_student_uml_edu/Stanford_valiData_Abuzz.csv"
    
    vali_data = ListDataset(v_csv, label_list, 512, 108)

    print('loading testing data')
    test_loader = DataLoader(vali_data, shuffle=False)

    out=[]
    y=[]
    for i, data in enumerate(test_loader):
        x, gt = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)
        with torch.no_grad():
            predicted = model(x)
            y.append(gt)
        out.append(predicted)
    
    yo=[]
    yp=[]
    c=0
    for i in range(len(out)):
        val = np.argmax(out[i].cpu().numpy()[0])
        val_ = y[i].tolist()[0]
        if val!=val_:
            c+=1
        yo.append(val_)
        yp.append(val)
    yo = np.array(yo)
    yp = np.array(yp)
    
    print(f'The total number of wrongly predicted samples out of total {len(out)} samples are: {c}')
    
    a = confusion_matrix(yo, yp)
    plotConfusionMatrix(a)
