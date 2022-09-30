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
  sns.heatmap(conf_matrix_df, annot=True, cmap=sns.color_palette("viridis", as_cmap=True))
  plt.title(title)
  plt.ylabel('True values')
  plt.xlabel('Predicted Values')
  plt.show()

  
# treniranje jedne epohe
def train_epoch(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
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

        for batch, (X, y) in enumerate(dataloader):
      
            X, y = X.to(device), y.to(device)
            pred_logits = model(X)

            loss = loss_fn(pred_logits, y)
            test_loss += loss.item()

            pred_labels = pred_logits.argmax(dim=1)
            test_acc += ((pred_labels == y).sum().item()/len(pred_labels))
            
            # rješenje "tensor ili numpy array" zavrzlame
            if device == 'cuda':
              epoch_true_labels.append(y.cpu())
              epoch_pred_labels.append(pred_labels.cpu())
            else:
              epoch_true_labels.append(y)
              epoch_pred_labels.append(pred_labels)
              
    epoch_true_labels = np.concatenate(epoch_true_labels)
    epoch_pred_labels = np.concatenate(epoch_pred_labels)
      
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc, epoch_true_labels, epoch_pred_labels

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

    roc_auc = {}
    best_accuracy = 0; # najbolji acc 
    es_counter = 1; # brojač epoha bez porasta acc 
    best_epoch = 0 # epoha s najboljim acc

    epochs_true, epochs_pred = [], [] # liste true/pred iz svake epohe
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_epoch(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss, test_acc, epoch_true, epoch_pred = test_epoch(model=model,
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

            print(classification_report(y_true=epoch_true, y_pred=epoch_pred, target_names=labels, zero_division=0))

            roc_auc = roc_auc_score_mc(epoch_true, epoch_pred, average = 'macro')
            print('{:>12}  {:>9}'.format("", "ROC_AUC (OvR)"))

            for l , v in roc_auc.items(): 
                print ('{:>12}  {:>9}'.format(labels[l], round(v, 4)))
        else:
            es_counter = es_counter + 1      

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        epochs_true.append(epoch_true)
        epochs_pred.append(epoch_pred)

        # provjera za early stopping
        if es_patience > 0:
          if es_counter > es_patience:
            print ('Test accuracy not improving for ', es_patience ,' epochs - early stopping.')
            print ('Best model saved in ', best_model)
            print ('Best test accuracy: ', best_accuracy)
            break
        
    return results, best_epoch, epochs_true, epochs_pred

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

    epochs_pred, epochs_true = [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # 0 = no early stopping
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    return results, best_epoch_true, best_epoch_pred

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

    epochs_pred, epochs_true = [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch_true, best_epoch_pred

  
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

    epochs_pred, epochs_true = [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch_true, best_epoch_pred

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
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=0.1, lr=0.1)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
   
    epochs_pred, epochs_true = [], []
    best_epoch = 0 

    # fine-tuning modela na našem skupu podataka
    results, best_epoch, epochs_true, epochs_pred = train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
    
    return results, best_epoch_true, best_epoch_pred
