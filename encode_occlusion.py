import torch
import torch.nn.functional as F
from torch import nn


class OcclusionEncoder(nn.Module):
    def __init__(self, occlusion_modality, patch_size, batch_size):
        super().__init__()
        self.conv_tabular = nn.Sequential(
            nn.Conv2d(occlusion_modality, 64, kernel_size=3, padding=1, bias=False)
        )

        self.patch_size = patch_size
        self.batch_size = batch_size

    def hierarchical_upsample(self, input, scale_factor):
        """Hierarchically upsamples the input tensor."""
        temp_scale = 1.0
        output = input
        while temp_scale < scale_factor:
            temp_scale *= 2
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True
            )
            if temp_scale * 2 > scale_factor:
                remaining_scale = scale_factor / temp_scale
                output = F.interpolate(
                    output,
                    scale_factor=remaining_scale,
                    mode="bilinear",
                    align_corners=True,
                )
                break
        return output

    def forward(self, importances):
        """Forward pass for OcclusionEncoder."""
        # Ensure the tensor is on the same device as the model
        device = next(self.parameters()).device
        importances = importances.to(device)

        unsq_importances = importances.unsqueeze(-1).unsqueeze(-1)
        up_impor = self.hierarchical_upsample(unsq_importances, self.patch_size)
        encoded_conv_impor = self.conv_tabular(up_impor)
        return encoded_conv_impor
