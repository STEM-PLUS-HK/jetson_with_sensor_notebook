import torch
import torchvision


def create_resnet18_sensor_fusion(output_dim, pretrained=True):
    """Stock ResNet18 with a late-fusion ToF head.

    Original road-following model:
        model = resnet18(...)
        model.fc = Linear(512, output_dim)   # 512 image features -> (x, y)

    Sequential cannot take a second input, so the old fc is split:
    Identity keeps 512, sensors are embedded, then concat + Linear.
    """
    # RESNET 18
    model = torchvision.models.resnet18(pretrained=pretrained)
    # Identity: keep the 512-d pooled vector. Linear(512, 2) would already
    # be (x, y) and ToF could not be concatenated onto the image embedding.
    model.fc = torch.nn.Identity()
    # 2 ToF values already mapped: 0-500 mm clamped, then 1-0 (closer is larger)
    model.sensor_fc = torch.nn.Linear(2, 16)
    # concat(512, 16) = 528 -> (x, y); this is the old fc, moved after concat
    model.fc_out = torch.nn.Linear(512 + 16, output_dim)

    _resnet_forward = model.forward  # body + Identity -> (B, 512)

    def forward(x, sensors):
        # dim=1 = feature axis: (B, 512) + (B, 16) -> (B, 528)
        x = torch.cat((_resnet_forward(x), model.sensor_fc(sensors)), dim=1)
        return model.fc_out(x)

    model.forward = forward  # model(image, sensors) instead of model(image)
    return model
