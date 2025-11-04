class myGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(myGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out
