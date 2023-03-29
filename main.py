import torch
import torchvision
from src.dataset import PennFudanDataset
from src.engine import train_model

# Definir as transformações a serem aplicadas às imagens e aos alvos
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Criar o conjunto de dados de treinamento
train_dataset = PennFudanDataset(root='dataset', transforms=transform)

# Definir o modelo
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Definir os parâmetros de treinamento
batch_size = 2
num_epochs = 10
learning_rate = 0.005

# Treinar o modelo
train_model(model, train_dataset, batch_size, num_epochs, learning_rate)
