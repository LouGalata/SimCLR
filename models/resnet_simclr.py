import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=512, num_input_channel=3):
        super(ResNetSimCLR, self).__init__()
        pretrained = True
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=pretrained),
            #                 "resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}


        self.backbone = self._get_basemodel(base_model)
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if num_input_channel != 3:
            self.backbone.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.conv1.requires_grad_(True)



        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.out_dim = out_dim
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
