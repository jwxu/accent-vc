import torch
import torch.nn as nn
import torch.nn.functional as F


class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell,
                            num_layers=num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)


    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds_normalized = embeds.div(norm)
        return embeds_normalized


class D_ACCENT_VECTOR(nn.Module):
    """d vector accent embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64, label_dim=6, classification=False):
        super(D_ACCENT_VECTOR, self).__init__()
        self.n_classes = label_dim
        self.classification = classification
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.fc_out = nn.Linear(dim_emb, label_dim)
        
        
    def forward(self, x):
        batch_size = x.size(0)

        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)

        if self.classification:
            logits = self.fc_out(embeds_normalized)
            logits = logits.view(batch_size, self.n_classes)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs

        return embeds_normalized
    