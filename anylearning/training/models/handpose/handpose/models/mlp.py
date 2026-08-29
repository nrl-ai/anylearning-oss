import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_layers = cfg["models"]["arch"]["backbone"]["hidden_layers"]
        hidden_units = cfg["models"]["arch"]["backbone"]["hidden_units"]
        output_units = cfg["models"]["arch"]["head"]["output_units"]
        backbone_dropout_rate = cfg["models"]["arch"]["backbone"]["dropout_rate"]
        head_dropout_rate = cfg["models"]["arch"]["head"]["dropout_rate"]

        self.fcn = nn.ModuleList()
        for i in range(hidden_layers):
            if i == hidden_layers - 1:
                self.fcn.append(
                    nn.ModuleList(
                        [
                            nn.Linear(hidden_units[i], hidden_units[i]),
                            nn.BatchNorm1d(hidden_units[i]),
                        ]
                    )
                )
            else:
                self.fcn.append(
                    nn.ModuleList(
                        [
                            nn.Linear(hidden_units[i], hidden_units[i + 1]),
                            nn.BatchNorm1d(hidden_units[i + 1]),
                        ]
                    )
                )
        self.outputs = nn.Linear(hidden_units[-1], output_units)
        self.backbone_dropout = nn.Dropout(backbone_dropout_rate)
        self.head_dropout = nn.Dropout(head_dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for f in self.fcn:
            x = f[0](x)  # Linear layer first
            # BatchNorm1d cannot compute batch statistics from a single sample,
            # so it is skipped for a batch of one -- but only while training. In
            # eval mode it uses its running statistics and is always safe, and
            # skipping it there meant ONNX export, which traces a batch of one,
            # baked out every batch norm layer: the exported model was not the
            # model that was trained. Testing self.training first also keeps the
            # tensor out of the condition, so torch.jit.trace has no data
            # dependent branch to warn about.
            if not (self.training and x.size(0) == 1):
                x = f[1](x)  # Batch norm
            x = F.relu(x)
            x = self.backbone_dropout(x)

        x = self.outputs(x)
        x = self.head_dropout(x)
        return x
