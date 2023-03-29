import time


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter
        self.meters = {}

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter(self.delimiter)
        self.meters[name].update(val)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(metric_str)


class AverageMeter:
    def __init__(self, delimiter="\t"):
        self.delimiter = delimiter
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = "{avg" + self.delimiter + ".3f} ({val" + self.delimiter + ".3f})"
        return fmt_str.format(avg=self.avg, val=self.val)

def collate_fn(batch):
    return tuple(zip(*batch))
