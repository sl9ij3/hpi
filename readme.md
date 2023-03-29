Aqui está a estrutura do meu projeto:
`
projeto/
├── dados/
│   ├── Annotation/
│   ├── Masks/
│   ├── PNGImages/
├── pesos/
├── resultados/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── model.py
│   └── utils.py
└── main.py
`;

main.py:
`
import torch
from src.config import Config
from src.dataset import CarDataset
from src.engine import train_one_epoch, evaluate
from src.model import get_instance_segmentation_model
from src.utils import collate_fn

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Configurações do modelo e dataset
    config = Config()
    model = get_instance_segmentation_model(config.num_classes)
    dataset = CarDataset('dados/treino', 'dados/teste', split='train')
    dataset_test = CarDataset('dados/treino', 'dados/teste', split='test')

    # Otimizador e Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

    # Treinamento
    for epoch in range(config.num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, config.print_freq)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    # Salva os pesos do modelo
    torch.save(model.state_dict(), 'weights/modelo.pth')

if __name__ == '__main__':
    main()
`;

src/config.py:
`
import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description="Treinamento do modelo")

    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch para treinamento")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizagem")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas para treinamento")
    parser.add_argument("--save-dir", type=str, default="../pesos/", help="Diretório para salvar os pesos treinados")

    return parser.parse_args()
`;

src/dataset.py:
`
import os
import torch
from torch.utils.data import Dataset, DataLoader

class HPIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data = torch.load(data_path)
        return data

def get_data_loaders(train_dir, test_dir, batch_size):
    train_dataset = MyDataset(train_dir)
    test_dataset = MyDataset(test_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
`;

src/model.py:
`
import torchvision

class MaskRCNN(torchvision.models.detection.MaskRCNN):
    def __init__(self, config):
        # Configurar o backbone do modelo
        if config["backbone"] == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True).features
            backbone.out_channels = 2048
        else:
            raise ValueError(f"Backbone '{config['backbone']}' not supported")
        
        # Configurar a camada de RoIAlign
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )
        
        # Inicializar o modelo Mask R-CNN
        super().__init__(
            backbone,
            num_classes=config["num_classes"],
            box_roi_pool=roi_pooler,
            mask_roi_pool=roi_pooler
        )

def create_model(config):
    # Criar o modelo Mask R-CNN
    mask_rcnn = MaskRCNN(config)
    
    return mask_rcnn
`;

src/engine.py:
`
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_dataset, batch_size, num_epochs, learning_rate):
    # Definir o otimizador e a função de perda
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Criar o DataLoader para carregar os dados de treino em lotes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Laço de treinamento
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Obter as entradas e as saídas esperadas do lote atual
            inputs, labels = data

            # Zerar os gradientes dos parâmetros do modelo
            optimizer.zero_grad()

            # Passar as entradas pelo modelo para obter as saídas
            outputs = model(inputs)

            # Calcular a perda para o lote atual
            loss = criterion(outputs, labels)

            # Calcular os gradientes da perda em relação aos parâmetros do modelo
            loss.backward()

            # Atualizar os parâmetros do modelo
            optimizer.step()

            # Acumular a perda para exibir estatísticas ao final do época
            running_loss += loss.item()

        # Exibir as estatísticas ao final de cada época
        print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # Salvar os pesos do modelo
    torch.save(model.state_dict(), 'pesos/model.pth')

def test_model(model, test_loader, device):
    """
    Testa um modelo em um conjunto de dados de teste.

    Args:
        model: modelo treinado a ser testado.
        test_loader: dataloader contendo o conjunto de dados de teste.
        device: dispositivo (CPU ou GPU) onde o modelo será executado.

    Retorna:
        Tuple contendo a acurácia média do modelo e as previsões do modelo para todo o conjunto de teste.
    """
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions += pred.cpu().numpy().tolist()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, predictions

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
`.