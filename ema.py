import torch
import copy

class EMA:
    def __init__(self, model, beta=0.995):
        self.beta = beta
        self.model = model
        self.ema_model = copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    v.copy_(self.beta * v + (1. - self.beta) * msd[k])
