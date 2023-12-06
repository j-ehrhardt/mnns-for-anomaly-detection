import torch

# Loss functions
def l1_loss(y, y_hat):
    return torch.mean(y - y_hat)

def l2_loss(y, y_hat):
    return torch.mean((y - y_hat)**2)

def rmse_loss(y, y_hat):
    return torch.mean(torch.sqrt((y - y_hat)**2))

def kl_div_loss(z, device):
    target = torch.normal(mean=0, std=1, size=z.shape).to(device)
    kl_div = torch.nn.KLDivLoss()
    return kl_div(z, target)

