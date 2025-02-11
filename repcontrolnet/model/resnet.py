
import torch
import torch.utils.checkpoint
import copy

from diffusers.utils import logging, deprecate
from diffusers.models.resnet import (  # noqa
    ResnetBlock2D,
)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class RepResnetBlock2D(ResnetBlock2D):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1_rep = copy.deepcopy(self.conv1)
        self.conv2_rep = copy.deepcopy(self.conv2)
        if self.use_in_shortcut:
            self.conv_shortcut_rep = copy.deepcopy(self.conv_shortcut)
            
    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
        rep_hidden_states = self.conv1_rep(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = hidden_states + rep_hidden_states

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        rep_hidden_states = self.conv2_rep(hidden_states) 
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states + rep_hidden_states

        if self.conv_shortcut is not None:
            input_tensor_rep = self.conv_shortcut_rep(input_tensor)
            input_tensor = self.conv_shortcut(input_tensor)
            input_tensor = input_tensor + input_tensor_rep

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
