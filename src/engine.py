import torch
import utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Epoch: [{}]'.format(epoch)
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(losses).all(), "Loss is not finite. {} {}".format(losses, loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        metric_logger.update(loss=losses, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['batch_time'].update(batch_time)
        metric_logger.meters['img/s'].update(images[0].size(0) / batch_time)

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        res = {target["image_id"].item(): output[i] for i, target in enumerate(targets)}
        evaluator = targets[0]["bbox"].metric_evaluator()
        evaluator.add(res)
        metric_logger.update(evaluator=evaluator)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    evaluator = metric_logger.meters['evaluator']
    return {k: v for k, v in evaluator.coco_eval.items()}

