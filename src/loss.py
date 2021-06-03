import torch.nn as nn
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), f'Pred shape {y_pred.shape} does not match target shape {y_true.shape}'
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class WeightedBCELoss(nn.Module):
    
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, y, y_true):
        loss = self.weight * y_true * self.bce(y)
        return torch.sum(loss) / loss.shape[0]


class DistanceMapLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DistanceMapLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, x, y, distance_map):
        assert x.size() == y.size()\
            and y.size() == distance_map.size(),\
            f'x shape {x.shape}, y shape {y.shape}, and map shape {map.shape} incompatible'

        loss = self.bce(x, y)
        loss *= distance_map + self.smooth
        return torch.sum(loss) / loss.shape[0]


class DMDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DMDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss(reduction='none')
        self.dsc_loss = DiceLoss()

    def forward(self, x, y, distance_map):
        assert x.size() == y.size()\
            and y.size() == distance_map.size(),\
            f'x shape {x.shape}, y shape {y.shape}, and map shape {map.shape} incompatible'

        loss = self.bce(x, y)
        loss *= distance_map + self.smooth
        return torch.sum(loss) / loss.shape[0]
