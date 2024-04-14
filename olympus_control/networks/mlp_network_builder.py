import torch
from torch import nn


class MLPBuilder(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_sizes,
        output_size,
        pth_file: str,
        device: str
    ) -> None:
        super().__init__()

        self._pth = torch.load(pth_file, map_location=torch.device('cpu'))
        self._weights = self._pth["model"]
        self._num_mlp_layers = 3


        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ELU())
        self.hidden_layers = nn.Sequential(*layers)

        j = 0
        for layer in self.hidden_layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = self._weights["a2c_network.actor_mlp.{}.weight".format(j)]
                layer.bias.data = self._weights["a2c_network.actor_mlp.{}.bias".format(j)]
                j+=2

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.output_layer.weight.data = self._weights["a2c_network.mu.weight"]
        self.output_layer.bias.data = self._weights["a2c_network.mu.bias"]
        self.register_buffer("_obs_mean", self._weights["running_mean_std.running_mean"].unsqueeze(0))
        self.register_buffer("_obs_std", self._weights["running_mean_std.running_var"].sqrt().unsqueeze(0))
        self._model = nn.Sequential(self.hidden_layers, self.output_layer)
        self._model = self._model.to(device=device)
        self._model.eval()

    def forward(self, obs):
        norm_obs = ((obs - self._obs_mean) / self._obs_std).to(torch.float32)
        norm_obs = norm_obs.squeeze(0)
        action = self._model(norm_obs)
        action = torch.clamp(action, -1.0, 1.0)
        return action



