from torch import nn, optim

class VolAutoEncoder(nn.Module):
    """
       This is the standard way to define your own network in PyTorch. You typically choose the components
       (e.g. LSTMs, linear layers etc.) of your network in the __init__ function.
       You then apply these layers on the input step-by-step in the forward function.
       You can use torch.nn.functional to apply functions such as F.relu, F.sigmoid, F.softmax.
       Be careful to ensure your dimensions are correct after each step.
    """
    # 
    def __init__(self):
        super(VolAutoEncoder, self).__init__()
        
        self.encoder=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(1, 64, (15, 15, 15), stride=3), # 36 36 36
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 256, (12, 12, 12), stride=4),
            nn.ReLU(inplace=True),
        )
        # 256 7 7 7

        self.linear = nn.Sequential(
            nn.Linear(87808, 87808),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 64, (9, 9, 9), stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 3, (20, 20, 20), stride=5),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.
        """
        encoded = self.encoder(x)
        y = encoded.reshape(-1,87808)
        z = self.linear(y)
        w = z.reshape((-1, 256, 7, 7, 7))
        decoded = self.decoder(w)
        r = decoded.reshape(-1, 3, 120, 120, 120)  # 120*120*120=1728000
        # out = self.sigmoid(r)
        out = self.softmax(r)

        return out
