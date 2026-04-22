import torch
import torch.nn as nn


# =========================================
# 🔥 LSTM MODEL CLASS
# =========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=126, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully Connected Layer
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_size)
        """

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take last time step
        out = out[:, -1, :]

        # Final output
        out = self.fc(out)

        return out


# =========================================
# 🔥 BUILD MODEL (for training if needed)
# =========================================
def build_model():
    model = LSTMModel(
        input_size=5,
        hidden_size=126,
        num_layers=3,
        dropout=0.2
    )
    return model


# =========================================
# 🔥 SAVE MODEL
# =========================================
def save_model(model, path="lstm_model.pth"):
    torch.save(model.state_dict(), path)


# =========================================
# 🔥 LOAD MODEL (FOR DEPLOYMENT)
# =========================================
def load_model(path="lstm_model.pth"):
    model = LSTMModel(
        input_size=5,
        hidden_size=126,
        num_layers=3,
        dropout=0.2
    )

    # Load weights
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))
    )

    model.eval()
    return model


# =========================================
# 🔥 PREDICTION FUNCTION (IMPORTANT)
# =========================================
def predict(model, input_data, scaler_x=None, scaler_y=None):
    """
    input_data shape: (sequence_length, 5)
    """

    model.eval()

    # Convert to numpy if needed
    if not isinstance(input_data, torch.Tensor):
        import numpy as np
        input_data = np.array(input_data)

    # Scale input
    if scaler_x is not None:
        input_data = scaler_x.transform(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    output = output.numpy()

    # Inverse scale output
    if scaler_y is not None:
        output = scaler_y.inverse_transform(output)

    return float(output[0][0])