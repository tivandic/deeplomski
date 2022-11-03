import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
import os
import glob
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, classification_report
import pandas as pd
import numpy as np
import random
from PIL import Image 
from skimage.io import imread
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.utils import shuffle
from skimage.filters import sobel, prewitt
import cv2
from skimage.filters import sobel, prewitt
from sklearn.utils import shuffle
from sklearn import preprocessing
from skimage.feature import hog
from skimage.feature import local_binary_pattern

from torch.optim.lr_scheduler import ExponentialLR

# učitava podatke za RF_SVM podijeljene na train/test
def LoadData(train_dir, test_dir, dir_separator, file_extension, image_size):
 
    train_images = []
    train_labels = [] 

    print('Učitavanje trening skupa...')
    for directory_path in tqdm(glob.glob(train_dir + "*")):
        label = directory_path.split(dir_separator)[-1]
        print(label)
        for img_path in tqdm(glob.glob(os.path.join(directory_path, file_extension))):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.resize(img, (image_size, image_size))
            
            # normalizacija na raspon od 0 do 255
            norm = np.zeros((image_size,image_size))
            img = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
            norm = None

            train_images.append(img)
            train_labels.append(label)
        
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images, train_labels = shuffle(train_images, train_labels)

    test_images = []
    test_labels = [] 

    print('Učitavanje test skupa...')
    for directory_path in tqdm(glob.glob(test_dir + "*")):
        label = directory_path.split(dir_separator)[-1] 
        print(label)
        for img_path in tqdm(glob.glob(os.path.join(directory_path, file_extension))):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size, image_size))

            # normalizacija na raspon od 0 do 255
            norm = np.zeros((image_size,image_size))
            img = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
            norm = None
            
            test_images.append(img)
            test_labels.append(label)
        
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
        
    return train_images, train_labels, test_images, test_labels

# učitava podatke i dijeli ih na train/test    
def LoadSplitData(data_dir, dir_separator, file_extension, image_size):
    
    from sklearn.model_selection import train_test_split

    images = []
    labels = [] 

    print('Učitavanje podataka...')
    for directory_path in tqdm(glob.glob(data_dir + "*")):
        label = directory_path.split(dir_separator)[-1]
        print(label)
        for img_path in tqdm(glob.glob(os.path.join(directory_path, file_extension))):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.resize(img, (image_size, image_size))
            
            # normalizacija na raspon od 0 do 255
            norm = np.zeros((image_size,image_size))
            img = cv2.normalize(img,  norm, 0, 255, cv2.NORM_MINMAX)
            norm = None
    
            images.append(img)
            labels.append(label)
        
    images = np.array(images)
    labels = np.array(labels)

    #images, labels = shuffle(images, labels)

    # dijeljenje na testni i trening skup
    (train_images, test_images, train_labels, test_labels) = train_test_split(images, labels, test_size = .2, random_state = 42)
        
    return train_images, train_labels, test_images, test_labels

# ekstrakcija značajki sobel/prewitt
def feature_extractor(dataset):
    train_images = dataset
    image_dataset = pd.DataFrame()
    i = 0
    for image in tqdm(range(train_images.shape[0])):  #iterate through each file 
        
        df = pd.DataFrame()          
        input_img = train_images[image, :,:,:]
        
        # gaussian blur
        img = input_img
        img = cv2.GaussianBlur(img, (3,3), 0)
         
        # sobel
        edge_sobel = sobel(img)
        edge_sobel_re = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel_re

        # prewitt
        edge_prewitt = prewitt(img)
        edge_prewitt_re = edge_prewitt.reshape(-1)
        df['Prewitt'] = edge_prewitt_re

        image_dataset = image_dataset.append(df)
        
    return image_dataset

# ekstrakcija značajki HOG/LBP
def f_extractor(dataset):
    
    from skimage.feature import hog
    from skimage import feature
    #import mahotas as mt
    
    train_images = dataset
    image_dataset = pd.DataFrame()
    i = 0
    for image in tqdm(range(train_images.shape[0])):  #iterate through each file 
        
        df = pd.DataFrame()          
        img = train_images[image, :,:,:]
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        radius = 3
        # number of neighbors to consider for LBP
        n_points = 8 * radius 
        # sampling type for LBP
        METHOD = 'uniform'     
        lbp = local_binary_pattern(img, n_points, radius, METHOD)
        # Converting into 1-D array
        f_lbp=lbp.flatten()
        
        f_hog, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
        
        feat = np.hstack([f_lbp, f_hog])
        
        df['features'] = feat
        
        image_dataset = image_dataset.append(df)
        
    return image_dataset
  
# prikazuje 16 nasumičnih slika iz zadanog skupa
def display_random_images(train_dir:str):
  images = []
  for dir in os.listdir(train_dir):
    for image in os.listdir(train_dir + '/' + dir):
      images.append(os.path.join(train_dir, dir, image))

  plt.figure(1, figsize=(15, 9))
  plt.axis('off')
  n = 0
  for i in range(16):
    n += 1
    random_img = random.choice(images)
    imgs = imread(random_img)
    plt.subplot(4, 4, n)
    plt.imshow(imgs)
  
  plt.show()

# prikazuje bar graph s brojem primjeraka u svakoj klasi
def classes_bar_plot(train_dir, extension):
  classes = {}
  num_clasess = 0
  for directory_path in glob.glob(train_dir):
    label = directory_path.split("/")[-1]
    #print(label)
    num_clasess = num_clasess + 1
    num_images = 0
    for img_path in glob.glob(os.path.join(directory_path, extension)):
        #print(img_path)
        num_images = num_images + 1
    classes[label] = num_images
    # print("Images in class " + label + ": " + str(num_images))    
  #print(classes)
  names = list(classes.keys())
  values = list(classes.values())
  plt.bar(range(len(classes)), values, tick_label=names)
  plt.xticks(rotation='vertical')
  plt.show() 
  
  
# roc_auc multi class OvR
def roc_auc_score_mc(true_class, pred_class, average):
  classes = set(true_class)
  roc_auc_dict = {}
  for one_class in classes:
    other_classes = [x for x in classes if x != one_class]
    true_classes = [0 if x in other_classes else 1 for x in true_class]
    pred_classes = [0 if x in other_classes else 1 for x in pred_class]
    roc_auc = roc_auc_score(true_classes, pred_classes, average = average)
    roc_auc_dict[one_class] = roc_auc

  return roc_auc_dict


# roc_auc_curve multi class OvR
def roc_auc_curve_mc(true_class, pred_class, classes):

  from sklearn import metrics

  j_classes = set(true_class)
  roc_auc_dict = {}
  for one_class in j_classes:
    other_classes = [x for x in j_classes if x != one_class]
    true_classes = [0 if x in other_classes else 1 for x in true_class]
    pred_classes = [0 if x in other_classes else 1 for x in pred_class]

    fpr, tpr, threshold = metrics.roc_curve(true_classes, pred_classes)
    roc_auc = metrics.auc(fpr, tpr)

    plt.rcParams["figure.figsize"] = (12,10)
    plt.title(classes[one_class])
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# loss/accuracy per epoch krivulje
def plot_results(results):

    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.plot(epochs, train_accuracy, label="Train accuracy")
    plt.plot(epochs, test_accuracy, label="Test accuracy")
    plt.legend()

# matrica konfuzije
def con_matrix(best_true, best_pred, class_names, title):
  conf_matrix = confusion_matrix(best_true, best_pred)
  conf_matrix_df = pd.DataFrame(conf_matrix,
                     index = class_names, 
                     columns = class_names)

  plt.figure(figsize=(12,10))
  sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap=sns.color_palette("viridis", as_cmap=True))
  plt.title(title)
  plt.ylabel('True values')
  plt.xlabel('Predicted Values')
  plt.show()

  
# treniranje jedne epohe
def train_epoch(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler, # 03-11-22 
               device: torch.device) -> Tuple[float, float]:

    train_loss, train_acc = 0, 0
   

   
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    scheduler.step()
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# testiranje jedne epohe
def test_epoch(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    test_loss, test_acc = 0, 0
    model.eval()     
    
    with torch.inference_mode():
  
        epoch_true_labels = []
        epoch_pred_labels = []
        epoch_probs = []

        for batch, (X, y) in enumerate(dataloader):
      
            X, y = X.to(device), y.to(device)
            pred_logits = model(X)

            loss = loss_fn(pred_logits, y)
            test_loss += loss.item()

            pred_labels = pred_logits.argmax(dim=1)
            test_acc += ((pred_labels == y).sum().item()/len(pred_labels))
            
            #probs = pred_logits.softmax(dim=-1).detach().cpu().flatten().numpy()
            probs = pred_logits.softmax(dim=-1).detach().cpu().numpy()
            epoch_probs.append(probs)
            
            # rješenje "tensor ili numpy array" zavrzlame
            if device == 'cuda':
              epoch_true_labels.append(y.cpu())
              epoch_pred_labels.append(pred_labels.cpu())
            else:
              epoch_true_labels.append(y)
              epoch_pred_labels.append(pred_labels)

    epoch_probs = np.concatenate(epoch_probs)
    epoch_true_labels = np.concatenate(epoch_true_labels)
    epoch_pred_labels = np.concatenate(epoch_pred_labels)

    roc_auc = metrics.roc_auc_score(epoch_true_labels, epoch_probs, multi_class='ovr', average = 'macro')
    #print('Epoch roc_auc_score: '+ str(roc_auc))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc, epoch_true_labels, epoch_pred_labels, roc_auc

# treniranje modela; nova funkcija
def train_model(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          es_patience: int,
          best_model: str,
          labels: list,
          device: torch.device) -> Dict[str, List]:

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
    } # rezultati

    # dodavano 3-11-22
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
    
    roc_auc = {}
    best_accuracy = 0; # najbolji acc 
    es_counter = 1; # brojač epoha bez porasta acc 
    best_epoch = 0 # epoha s najboljim acc

    epoch_aucs, epochs_true, epochs_pred = [], [], [] # liste true/pred/prob iz svake epohe
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_epoch(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           scheduler=scheduler, 
                                           device=device)
        
        test_loss, test_acc, epoch_true, epoch_pred, epoch_auc = test_epoch(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)


        print ('============================================================================================')
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} "
        )
        
   
        
        if    test_acc > best_accuracy: 
            torch.save(model.state_dict(), best_model)
            print('---------------------------------------------------------------------------------------------')
            print('Epoch ', (epoch + 1), '| Best model saved in ', best_model) 
            best_accuracy = test_acc
            es_counter = 1
            best_epoch = epoch

            #print(classification_report(y_true=epoch_true, y_pred=epoch_pred, target_names=labels, zero_division=0))
            
            #roc_auc = roc_auc_score_mc(epoch_true, epoch_pred, average = 'macro')
            #print('{:>12}  {:>9}'.format("", "ROC_AUC (OvR)"))

            #for l , v in roc_auc.items(): 
             #   print ('{:>12}  {:>9}'.format(labels[l], round(v, 4)))
        else:
            es_counter = es_counter + 1      

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        epochs_true.append(epoch_true)
        epochs_pred.append(epoch_pred)
        epoch_aucs.append(epoch_auc)
        
        # provjera za early stopping
        if es_patience > 0:
          if es_counter > es_patience:
            print ('Test accuracy not improving for ', es_patience ,' epochs - early stopping.')
            print ('Best model saved in ', best_model)
            print ('Best test accuracy: ', best_accuracy)
            break
        
    return results, best_epoch, epochs_true, epochs_pred, epoch_aucs
  
# ===========================
# funkcije za deep modele
#============================

# Alexnet
def DAlexNet (train_dir, test_dir, model_path):
  
    # postavlja izvršavanje na GPU ili CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.AlexNet_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
    
    # učitavanje podataka u dataloader
    train_data = datasets.ImageFolder(train_dir, transform = preprocess)
    test_data = datasets.ImageFolder(test_dir, transform = preprocess)

    class_names = train_data.classes

    train_dataloader = DataLoader(
      train_data,
      batch_size=32,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      )
    test_dataloader = DataLoader(
      test_data,
      batch_size = 32,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
      )
      
    model = torchvision.models.alexnet(weights=weights).to(device)
   
    # "zamrzavamo" sve trainable slojeve osim klasifikatora
    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # kreiramo novi klasifikator 
    model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5), # vrijednost p prema Krizhevsky et al. 2012
    torch.nn.Linear(in_features=9216, # dimenzije izlaza prethodnog sloja
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5), # vrijednost p prema Krizhevsky et al. 2012
    torch.nn.Linear(in_features=4096, # dimenzije izlaza prethodnog sloja
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=4096, # dimenzije izlaza prethodnog sloja
                    out_features=len(class_names), # dimenzije izlaznog sloja = broj klasa u skupu podataka
                    bias=True)).to(device)
    
    # standardni optimizer i loss function za AlexNet
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs_pred, epochs_true, epoch_aucs = [], [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred, epoch_aucs = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 3, # 0 = no early stopping
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    return results, best_epoch, epochs_true, epochs_pred, epoch_aucs

# ResNet50
def DResNet50 (train_dir, test_dir, model_path):
  
    # postavlja izvršavanje na GPU ili CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.ResNet50_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
    
    # učitavanje podataka u dataloader
    train_data = datasets.ImageFolder(train_dir, transform = preprocess)
    test_data = datasets.ImageFolder(test_dir, transform = preprocess)

    class_names = train_data.classes

    train_dataloader = DataLoader(
      train_data,
      batch_size=32,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      )
    test_dataloader = DataLoader(
      test_data,
      batch_size = 32,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
      )
    
    model = torchvision.models.resnet50(weights=weights).to(device)
    
    # "zamrzavamo" sve trainable slojeve osim klasifikatora
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False    
    for param in model.layer2.parameters():
        param.requires_grad = False    
    for param in model.layer3.parameters():
        param.requires_grad = False    
    for param in model.layer4.parameters():
        param.requires_grad = False    
 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # kreiramo novi klasifikator 
    model.fc = torch.nn.Linear(in_features=2048, 
                    out_features=len(class_names), 
                    bias=True).to(device)
    
    # standardni optimizer i loss function za ResNET
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs_pred, epochs_true, epoch_aucs = [], [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred, epoch_aucs = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 3, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch, epochs_true, epochs_pred, epoch_aucs

  
# VGG16
def DVGG16 (train_dir, test_dir, model_path):    
  
    # postavlja izvršavanje na GPU ili CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.VGG16_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
    
    # učitavanje podataka u dataloader
    train_data = datasets.ImageFolder(train_dir, transform = preprocess)
    test_data = datasets.ImageFolder(test_dir, transform = preprocess)

    class_names = train_data.classes

    train_dataloader = DataLoader(
      train_data,
      batch_size=32,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      )
    test_dataloader = DataLoader(
      test_data,
      batch_size = 32,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
      )

    model = torchvision.models.vgg16(weights=weights).to(device)

    # "zamrzavamo" sve trainable slojeve osim klasifikatora
    for param in model.features.parameters():
        param.requires_grad = False
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # kreiramo novi klasifikator 
    model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=25088, 
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(in_features=4096, 
                    out_features=4096, 
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(in_features=4096, 
                    out_features=len(class_names), 
                    bias=True)).to(device)
    
    # standardni optimizer i loss function za VGG16
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs_pred, epochs_true, epoch_aucs = [], [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred, epoch_aucs = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 3, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch, epochs_true, epochs_pred, epoch_aucs

# ViT_b_16
def DViT_16 (train_dir, test_dir, model_path):    
  
    # postavlja izvršavanje na GPU ili CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
    
    # učitavanje podataka u dataloader
    train_data = datasets.ImageFolder(train_dir, transform = preprocess)
    test_data = datasets.ImageFolder(test_dir, transform = preprocess)

    class_names = train_data.classes

    train_dataloader = DataLoader(
      train_data,
      batch_size=32,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      )
    test_dataloader = DataLoader(
      test_data,
      batch_size = 32,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
      )
    
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    
    # "zamrzavamo" sve trainable slojeve osim klasifikatora
    for param in model.conv_proj.parameters():
        param.requires_grad = False   
    for param in model.encoder.parameters():
        param.requires_grad = False
 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # kreiramo novi klasifikator 
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, 
                    out_features=len(class_names), 
                    bias=True)).to(device)                                    
    
    # standardni optimizer i loss function za ViT16
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=0.01, lr=0.001)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
   
    epochs_pred, epochs_true, epoch_aucs = [], [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred, epoch_aucs = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 3, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch, epochs_true, epochs_pred, epoch_aucs
