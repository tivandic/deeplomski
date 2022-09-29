def DAlexNet (train_dir, test_dir, train_dataloader, test_dataloader, model_path):
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.AlexNet_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
      
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
    results, best_epoch, epochs_true, epochs_pred = deepl_funkcije.train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # 0 = no early stopping
                       best_model = model_path,
                       labels=class_names,
                       device = device)


def DResNet50 (train_dir, test_dir, model_path):
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.ResNet50_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
          
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
    results, best_epoch, epochs_true, epochs_pred = deepl_funkcije.train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)

def DVGG16 (train_dir, test_dir, model_path):    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.VGG16_Weights.DEFAULT 
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()

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
    results, best_epoch, epochs_true, epochs_pred = deepl_funkcije.train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)

def DViT_16 (train_dir, test_dir, model_path):    
    
    # učitavanje najboljih težinskih vrijednosti dobivenih treniranjem na ImageNet skupu
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    
    # transformacija ulaznih podataka korištenjem istih parametara kao za ImageNet
    # budući da ćemo koristiti model prethodno treniran na tom skupu podataka
    preprocess = weights.transforms()
     
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
    results, best_epoch, epochs_true, epochs_pred = deepl_funkcije.train_model(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       optimizer = optimizer,
                       loss_fn = loss_fn,
                       epochs = 30,
                       es_patience = 10, # postaviti 0 za bez early stoppinga
                       best_model = model_path,
                       labels=class_names,
                       device = device)
