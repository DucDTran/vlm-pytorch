from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,  # size of embedding vectors
        intermediate_size=3072,  # size of linear layers of the feed forward network
        num_hidden_layers=12,  # number of hidden layers of the Vision Transformer
        num_attention_heads=12,  # number of attention heads in the Multi-head Attention
        num_channels=3,  # number of channels each image has (R, G, B)
        image_size=224,  # size of the image
        patch_size=16,  # number of patches from each image
        layer_norm_eps=1e-6,  # parameter of the layer normalization
        attention_dropout=0.0,  # parameter of the Multi-head Attention
        # number of output embeddings for each image the Multi-head Attention has
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'  # No padding is needed
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_position = self.num_patches
        self.position_embedding = nn.Embedding(
            self.num_position, self.embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_position).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        # [batch_size, channels, height, width]
        _, _, height, width = pixel_values.shape
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim,Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm_1 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm_2 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [Batch_size, Num_patches, Embed_dim]
        residual = hidden_states
        # [Batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm_1(hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm_2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layer_norm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values = [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dims]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(input_embeds=hidden_states)
        last_hidden_states = self.post_layer_norm(last_hidden_states)
        return last_hidden_states


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dims]
        return self.vision_model(pixel_values=pixel_values)
