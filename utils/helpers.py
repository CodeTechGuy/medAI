
import torch



def get_confidence(q_values):
    probs = torch.softmax(q_values, dim=0)
    return torch.max(probs).item()

def calculate_entropy(q_values):
    probs = torch.softmax(q_values, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
    return entropy.item()
