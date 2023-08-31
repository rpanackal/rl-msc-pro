from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
