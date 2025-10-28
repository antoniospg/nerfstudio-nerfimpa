# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Siren NeRF field"""

from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfimpa.fields.custom_vanilla_field import CustomVanillaField

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=True, bias=True):
        # sin(w0 * (W * x + b))
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(in_features)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = (6.0 / in_features) ** 0.5 / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

class SirenMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        skip_connections: Tuple[int, ...] = (4,),
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = set(skip_connections)

        layers = nn.ModuleList()
        current_in = in_dim

        for layer_idx in range(num_layers):
            is_first = (layer_idx == 0)
            if layer_idx in self.skip_connections:
                current_in += in_dim

            layers.append(
                SineLayer(
                    in_features=current_in,
                    out_features=layer_width,
                    w0=(w0_first if is_first else w0_hidden),
                    is_first=is_first,
                )
            )

            current_in = layer_width

        # No sine in the final layer
        final_layer = nn.Linear(current_in, layer_width)
        with torch.no_grad():
            bound = (6.0 / current_in) ** 0.5 / w0_hidden
            final_layer.weight.uniform_(-bound, bound)
            if final_layer.bias is not None:
                final_layer.bias.uniform_(-bound, bound)

        self.layers = layers
        self.final_layer = final_layer
        self.out_dim = layer_width

    def get_out_dim(self):
        return self.out_dim

    def forward(self, x):
        x0 = x
        h = x

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.skip_connections:
                h = torch.cat([h, x0], dim=-1)
            h = layer(h)

        # final layer
        if len(self.layers) in self.skip_connections:
            h = torch.cat([h, x0], dim=-1)
        h = self.final_layer(h)

        return h

class SirenField(CustomVanillaField):
    """NeRF Field

    Args:k
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = SirenMLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            w0_first=30.0,
            w0_hidden=1.0,
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

        if field_heads:
            self.mlp_head = SirenMLP(
                in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                skip_connections=(),
                w0_first=30.0,
                w0_hidden=1.0,
            )

        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore
