import torch
from torch.utils.data import DataLoader
import dataset
import utils
from engine import train_one_epoch, evaluate
import torchvision
import warnings
warnings.filterwarnings("ignore")

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def main():
    # use our dataset and defined transformations
    dataset_train = dataset.PennFudanDataset('../dataset', dataset.get_transform(train=True))
    dataset_val = dataset.PennFudanDataset('../dataset', dataset.get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-50:])

    # define training and validation data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes=2)

    # move model to the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # train the model for 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device)

if __name__ == "__main__":
    main()
