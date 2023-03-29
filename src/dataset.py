import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # carregar todas as imagens de uma vez
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # carrega a imagem como PIL Image
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # carrega a máscara como PIL Image
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        mask = Image.open(mask_path)

        # converte a máscara para array numpy (0 para fundo, 1 para pedestre)
        mask = np.array(mask)
        mask[mask > 0] = 1

        # coordenadas das pessoas na imagem
        obj_ids = np.unique(mask)
        # primeiro id é sempre zero, portanto remove-se
        obj_ids = obj_ids[1:]

        # dividir a máscara em máscaras separadas para cada objeto
        masks = mask == obj_ids[:, None, None]

        # obter as caixas delimitadoras para cada máscara
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # converter tudo em torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # há apenas um rótulo aqui, pois só há uma imagem no conjunto de dados
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # supõe que todas as instâncias são pedestres
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
