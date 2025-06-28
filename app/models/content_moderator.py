import torch
from torch import nn
import os
from app.configs import device


class ContentModeratorModel(nn.Module):
    def __init__(self, video_dim, audio_dim, title_dim, hidden_dim=128):
        super().__init__()

        self.video_net = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
        )
        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
        )
        self.title_net = nn.Sequential(
            nn.Linear(title_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, video_x, audio_x, title_x):
        v = self.video_net(video_x)
        a = self.audio_net(audio_x)
        t = self.title_net(title_x)
        x = torch.cat([v, a, t], dim=1)
        return self.classifier(x)


model = ContentModeratorModel(512, 20000, 384)
model.load_state_dict(
    torch.load(
        os.path.join("app", "models", "content_moderator.pth"),
        map_location=torch.device(device),
    )
)
model.to(device)
