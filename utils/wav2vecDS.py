import numpy as np
import torch
import torch.nn as nn


# Construct the model for mapping the wav2vec features to DS
class _Wav2vecDS(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(_Wav2vecDS, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# mapping the features from wav2vec to DS


class Wav2vecDS(nn.Module):
    def __init__(self):
        super(Wav2vecDS, self).__init__()
        self.input_dim = 29
        self.hidden_dim = 512
        self.pretrained_model = torch.load("./asserts/wav2vecDS.pt")

    # define the mapping function
    def mapping(self, array):
        model = _Wav2vecDS(self.input_dim, self.hidden_dim)
        model.load_state_dict(self.pretrained_model)
        input_data = torch.tensor(array)
        # Perform inference
        with torch.no_grad():
            output = model(input_data)
        return output.numpy()


# if __name__ == "__main__":

#     # Load the trained model
#     model_path = "./asserts/wav2vecDS.pt"
#     loaded_model = torch.load(model_path)
#     # Assuming fc1.weight corresponds to input_dim
#     print(f"input_dim = :{loaded_model['fc1.weight'].shape[1]}")  # 29
#     # hidden_dim = loaded_model['fc1.weight'].shape[0]  # Assuming fc1.weight corresponds to hidden_dim
#     print(f"hidden_dim = :{loaded_model['fc1.weight'].shape[0]}")  # 512
#     input_dim = 29
#     hidden_dim = 512
#     # Create the Wav2vecDS model using the retrieved dimensions
#     model = Wav2vecDS(input_dim, hidden_dim)
#     model.load_state_dict(loaded_model)

#     # Prepare input data for inference (replace this with your actual data)
#     # Replace with your input data
#     input_data = torch.tensor(np.load("./wav2vec/audio_features_hubert.npy"))

#     # Perform inference
#     with torch.no_grad():
#         output = model(input_data)

#     # Process or save the output as needed
#     # For example, if you want to convert the output back to a numpy array:
#     output_array = output.numpy()
