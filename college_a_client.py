import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PlacementModel(nn.Module):
    def __init__(self):
        super(PlacementModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_college_a():
    print("=== TRAINING LOCAL MODEL: COLLEGE A ===")

    # Synthetic dataset for College A (Higher focus on GRE/TOEFL)
    X = torch.rand(500, 5)
    y = ((X[:, 0] * 0.4 + X[:, 2] * 0.4 + X[:, 3] * 0.2) > 0.5).float().unsqueeze(1)

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = PlacementModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(15):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Save metadata payload
    payload = {
        "college_id": "College_A_Engineering",
        "weights": model.state_dict()
    }

    output_path = "college_a_metadata.pt"
    torch.save(payload, output_path)
    print(f"✓ Complete! Copy '{output_path}' to your flash drive.")


if __name__ == "__main__":
    train_college_a()