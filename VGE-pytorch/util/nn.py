import torch

def latent_kl(q, p):
    mean1 = q
    mean2 = p

    kl = 0.5 * torch.square(mean2 - mean1)
    kl = torch.sum(kl, dim=[1, 2, 3])
    kl = torch.mean(kl)
    return kl