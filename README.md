# PlantGuard: Plant Disease Detection

## Overview
**PlantGuard** is a Flask-based web application designed to identify plant diseases from images using a pre-trained deep learning model. It employs the **ResNet-18** architecture, fine-tuned on the **Plant Pathology dataset**, to classify multiple plant diseases effectively.

---

## Key Features
- **ResNet-18 Model**: A pre-trained convolutional neural network (CNN) optimized for image classification.
- **Flask Backend**: Simple and lightweight API for image upload and prediction.
- **Custom Dataset Handling**: Includes preprocessing, augmentation, and efficient DataLoader integration.
- **Mixed Precision Training**: Improves computation speed and memory efficiency.
- **Early Stopping**: Prevents overfitting by monitoring validation accuracy.

---

## [Dataset](https://www.kaggle.com/datasets/sorooshtootoonchian/plantpathology2020fgvc7resized "Kaggle link to dataset")
The **Plant Pathology dataset** is used for training and evaluation. Data augmentation techniques include:
- Resizing images to 128x128.
- Applying random horizontal and vertical flips.
- Introducing random rotations (up to 40 degrees).
- Modifying brightness, contrast, and saturation.

---

## Technology Stack
- **Frontend**: Flask Templates (HTML/CSS)
- **Backend**: Flask (Python)
- **Deep Learning Framework**: PyTorch
- **Model Architecture**: ResNet-18
- **Deployment**: Flask-based app (compatible with Docker or cloud services)

---

## Demo

https://github.com/user-attachments/assets/932fb1c0-4edf-4905-ab19-7c56fa4426bf

---

## Model Training Pipeline

### Data Augmentation:
Applied transformations for training and validation:
```python
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  
```

### Custom Dataset Class:
```python
class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx, 1:].values.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, label

```  

### Model Definition:
The ResNet-18 model is modified to handle 4 output categories:
```python
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

    def forward(self, x):
        return self.model(x)
```

### Training and Evaluation:
Optimizer: Adam  
Scheduler: ReduceLROnPlateau  
Loss Function: Binary Cross-Entropy with Logits  
Mixed Precision: Enabled via torch.cuda.amp  

### Clone the repository:
```bash
git clone https://github.com/your-username/PlantGuard.git
cd PlantGuard
```

### Run the Flask app:
```bash
python app.py
```

### Acknowledgments:
- **Dataset**: Plant Pathology dataset.
- **Model**: ResNet-18 by Microsoft Research.
- **Framework**: PyTorch.
