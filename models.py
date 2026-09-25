import math
import json

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from pathlib import Path

from albef.xbert import BertConfig, BertModel
from conv.implicit_autoencoder import ImplicitEncoder

try:
    from transformers import BertModel as StandardBertModel, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    StandardBertModel = None
    BertTokenizer = None
    print("Warning: transformers library not available. Text encoding will not work.")

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class TextEncoder(nn.Module):
    """
    BERT-based text encoder for processing text instructions.
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=768, freeze_bert=False):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for TextEncoder")
        

        # ALBEF's BertModel expects fusion_layer in config
        # Load config from pretrained model and add fusion_layer if missing
        from transformers.models.bert.configuration_bert import BertConfig as StandardBertConfig

        # Load standard config
        standard_config = StandardBertConfig.from_pretrained(model_name)
        # Create ALBEF config by copying standard config and adding ALBEF-specific attributes
        # Get all standard config attributes as dict
        config_dict = standard_config.to_dict()
        # Add ALBEF-specific attributes
        config_dict['fusion_layer'] = standard_config.num_hidden_layers  # Default: fusion at all layers
        config_dict['add_cross_attention'] = True
        config_dict['encoder_width'] = standard_config.hidden_size
        # Create ALBEF BertConfig
        albef_config = BertConfig(**config_dict)
        # Create ALBEF BertModel with the config
        self.bert = BertModel(albef_config)
        # Load pretrained weights from standard model
        if StandardBertModel is None:
            raise ImportError("Standard transformers BertModel is required for loading pretrained weights.")
        pretrained_model = StandardBertModel.from_pretrained(model_name)
        # Load weights (strict=False to allow missing fusion_layer-related params)
        self.bert.load_state_dict(pretrained_model.state_dict(), strict=False)
     
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.output_dim = output_dim
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer to match desired output dimension
        if output_dim != 768:  # BERT base has 768 hidden size
            self.projection = nn.Linear(768, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, texts, batch_size=None):
        """
        Args:
            texts: Can be:
                - A single string: "This is the instruction text" (will be converted to list)
                - A list of strings: ["Text 1", "Text 2", ...] where each string is a complete instruction
                  (one per batch item). Each string contains the entire instruction language.
            batch_size: Optional batch size for when texts is None
        
        Returns:
            text_features: [B, 512, output_dim] - all token embeddings (up to 512 tokens)
            attention_mask: [B, 512] - attention mask from tokenizer (1 for real tokens, 0 for padding)
        
        Note:
            The tokenizer accepts both single strings and lists of strings. When a list is provided,
            each string is tokenized independently (batch processing). The tokenizer automatically
            creates attention masks: 1 for real tokens, 0 for padding tokens.
        """
        # Handle empty or None texts
        if texts is None or (isinstance(texts, list) and len(texts) == 0):
            # Return zero features (will be handled by model)
            device = next(self.bert.parameters()).device
            if batch_size is None:
                batch_size = 1  # Default batch size
            attention_mask = torch.zeros(batch_size, 512, dtype=torch.long, device=device)
            return torch.zeros(batch_size, 512, self.output_dim, device=device), attention_mask
        
        # Ensure texts is a list
        if not isinstance(texts, list):
            texts = [texts]
        # Replace empty strings with a default token
        texts = [t if t and t.strip() else "[PAD]" for t in texts]
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding="max_length",  # Pad to max_length (512) for consistent output size
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        # Get attention mask from tokenizer (1 for real tokens, 0 for padding)
        attention_mask = encoded['attention_mask']  # [B, 512] - 1 for real tokens, 0 for padding
        
        # Get BERT embeddings
        with torch.set_grad_enabled(self.training):
            outputs = self.bert(**encoded)
            # Get all token embeddings (no pooling)
            token_embeddings = outputs.last_hidden_state  # [B, 512, 768]
        
        # Project to output dimension (apply per-token)
        B, L, _ = token_embeddings.shape  # L = 512
        token_embeddings_flat = token_embeddings.view(B * L, -1)  # [B*512, 768]
        text_features_flat = self.projection(token_embeddings_flat)  # [B*512, output_dim]
        text_features = text_features_flat.view(B, L, self.output_dim)  # [B, 512, output_dim]
        
        return text_features, attention_mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))




class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, dim=256, hidden_dim=48, extra_dims=1, use_pos_embedding=True):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        
        if use_pos_embedding:
            assert hidden_dim % 6 == 0, "hidden_dim must be divisible by 6"
            self.embedding_dim = hidden_dim
            freq_count = self.embedding_dim // 6
            e = torch.pow(2, torch.arange(freq_count)).float() * math.pi
            e = torch.stack(
                [
                    torch.cat([e, torch.zeros(freq_count), torch.zeros(freq_count)]),
                    torch.cat([torch.zeros(freq_count), e, torch.zeros(freq_count)]),
                    torch.cat([torch.zeros(freq_count), torch.zeros(freq_count), e]),
                ]
            )
            self.register_buffer("basis", e)
        else:
            self.embedding_dim = 0

        input_dim = self.embedding_dim + 3 + extra_dims
        # 7-layer MLP for point embedding
        layers = []
        num_layers = 2
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else dim, dim))
            if i < num_layers-1:  # No activation after last layer
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def embed(coords, basis):
        projections = torch.einsum("bnd,de->bne", coords, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, points):
        coords = points[..., :3]
        extras = points[..., 3:]
        extras = extras if extras.numel() > 0 else None
        
        if self.use_pos_embedding:
            harmonic = self.embed(coords, self.basis)
            concat = torch.cat([harmonic, coords], dim=-1)
        else:
            concat = coords
        
        if extras is not None:
            concat = torch.cat([concat, extras], dim=-1)
        return self.mlp(concat)


class PointTransformer(nn.Module):
    """
    Point transformer for query-based reconstruction.
    
    - Input points (with features) produce a latent code via encoder
    - Query points (coordinates only) interact with latent code to predict labels
    """

    def __init__(
        self,
        *,
        num_points: int,
        num_query_points: int,
        point_feature_dim: int,
        num_classes: int,
        num_latents: int = 512,
        depth: int = 4,
        dim: int = 512,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        decoder_ff: bool = True,
        use_cnn: bool = True,
        use_image: bool = False,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_query_points = num_query_points
        self.use_cnn = use_cnn
        self.use_image = use_image and use_cnn
        
        
        extra_dims = max(point_feature_dim - 3, 0)

        # Input point embedding (coordinates plus one label channel)
        self.point_embed = PointEmbed(dim=dim, hidden_dim=48, extra_dims=extra_dims, use_pos_embedding=False)
        
        # Query point embedding (coordinates only)
        self.query_embed = PointEmbed(dim=dim, hidden_dim=48, extra_dims=0, use_pos_embedding=False)
        
        if use_cnn:
            # Shape only: one channel of multi-class labels.
            # With image: channel 0 is those labels, channel 1 is the scan.
            in_channels = 2 if self.use_image else 1
            self.volume_encoder = ImplicitEncoder(
                in_channels=in_channels,
                num_channels=[32, 32, 64, 128],  # Feature pyramid channels (matching checkpoint)
                num_layers=2,  # Layers per ConvBlock
            )
            
            # MLP to process CNN features + query coordinates for query point encoding
            # Input: CNN features (from grid_sample) + normalized query coords
            # Output: Query point embeddings
            # Checkpoint uses: input_dim=227 (224 CNN features + 3 coords), 5 layers
            # 227 = 224 + 3, so CNN feature dim from checkpoint is 224
            # But checkpoint's last pyramid level is 128, so there might be a projection or concatenation
            # Based on checkpoint structure: 5-layer MLP with input 227
            # Now adding query_labels_input: 224 CNN + 3 coords + 1 label = 228
            cnn_feature_dim = 224  # Matching checkpoint (may involve feature projection/concatenation)
            self.query_point_encoding_mlp = nn.Sequential(
                nn.Linear(cnn_feature_dim + 3 + 1, dim),  # 228 -> 512 (224 CNN + 3 coords + 1 label)
                nn.GELU(),
                nn.Linear(dim, dim),  # 512 -> 512
                nn.GELU(),
                nn.Linear(dim, dim),  # 512 -> 512 (5 layers total: 0-4)
            )
        else:
            # No CNN: use simple MLP to process query coordinates only
            # Input: query coordinates (3 dims)
            # Output: Query point embeddings
            self.query_point_encoding_mlp = nn.Sequential(
                nn.Linear(3, dim),  # 3 -> 512
                nn.GELU(),
                nn.Linear(dim, dim),  # 512 -> 512
                nn.GELU(),
                nn.Linear(dim, dim),  # 512 -> 512
            )
            self.volume_encoder = None
        
        # Latent tokens for encoding input points
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, dim))
        
        # Text encoder for text instructions
        self.text_encoder = TextEncoder(output_dim=dim, freeze_bert=True)
        

        # Encoder: input points -> latent code
        # Use smaller hardcoded feedforward for cross_ff (512 -> 768 -> 512)
        cross_ff_net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
        )
        self.cross_attend = nn.ModuleList(
            [
                PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim=dim),
                PreNorm(dim, cross_ff_net),
            ]
        )


        # Create BertConfig for multimodal fusion
        bert_config_path = Path(__file__).resolve().parent / "albef" / "config_bert.json"
        
          
        with open(str(bert_config_path), 'r') as f:
            bert_config_dict = json.load(f)
        # Adapt config to match our dimension
        bert_config_dict['hidden_size'] = dim
        bert_config_dict['encoder_width'] = dim  # For cross-attention
        bert_config_dict['intermediate_size'] = dim * ff_mult
        bert_config_dict['num_attention_heads'] = heads
        bert_config_dict['num_hidden_layers'] = depth
        # CRITICAL: Set fusion_layer to 0 so ALL layers have cross-attention enabled
        # In ALBEF, cross-attention (fusion) is enabled when layer_num >= fusion_layer
        # - fusion_layer=0 means: ALL layers (0, 1, 2, ...) have cross-attention
        # - fusion_layer=6 means: Only layers 6, 7, 8, ... have cross-attention
        # For multi-modal mode, we want all layers to attend to text features (like ALBEF decoder)
        # ALBEF uses fusion_layer=0 with 6 layers for multimodal fusion
        bert_config_dict['fusion_layer'] = 0
        bert_config = BertConfig(**bert_config_dict)
       
        # ALBEF BertModel handles attention-mask extension and text cross-attention.
        # encoder_embeds passes latents directly and skips the token embedding layer.
        self.multimodal_encoder = BertModel(bert_config)
        print("Multimodal encoder initialized from cowtalk/albef/xbert.py")
        print(f"  Config: depth={depth}, fusion_layer={bert_config.fusion_layer} (all {depth} layers have cross-attention enabled)", flush=True)
        print(f"  Architecture matches ALBEF decoder: {depth} layers with fusion_layer=0", flush=True)
        

        # Decoder: query points attend to latent code
        self.query_cross_attend = PreNorm(dim, Attention(dim, dim, heads=heads, dim_head=dim_head), context_dim=dim)
        if decoder_ff:
            ff_block = lambda: nn.Sequential(
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, 512),
            )
            self.query_ff_layers = nn.ModuleList([ff_block() for _ in range(2)])
        else:
            self.query_ff_layers = None

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, input_points: torch.Tensor, query_points: torch.Tensor, texts, 
                modified_resized: torch.Tensor, query_coords_resized: torch.Tensor, query_labels_input=None):
        B = input_points.shape[0]
        
        # Encode input points to latent code
        input_embeddings = self.point_embed(input_points)  # [B, N, dim]
        
        # Initialize latents from tokens
        latents = repeat(self.latent_tokens, "n d -> b n d", b=B)  # [B, num_latents, dim]
        
        
        # Cross-attention: latents attend to input points
        cross_attn, cross_ff = self.cross_attend
        latents_immediate = cross_attn(latents, context=input_embeddings) + latents
        latents = cross_ff(latents_immediate) + latents_immediate
        
        # Store latents before multimodal encoder for L1 regularization
        latents_before_multimodal = latents.clone()
        
        # Multimodal fusion: fuse text features with latents using ALBEF encoder
        # Encode text instructions
        text_features, text_attention_mask = self.text_encoder(texts, batch_size=B)  # [B, 512, dim], [B, 512]
        # Create attention masks
        # For latents: all ones (all latent tokens are valid)
        latents_attention_mask = torch.ones(B, latents.size(1), dtype=torch.long, device=latents.device)
        
        # Use BertModel.forward() with encoder_embeds to properly handle attention mask extension
        # This ensures masks are extended correctly before being passed to encoder layers
        multimodal_output = self.multimodal_encoder(
            encoder_embeds=latents,  # [B, num_latents, dim] - bypasses embedding layer
            attention_mask=latents_attention_mask,  # [B, num_latents] - will be extended by BertModel
            encoder_hidden_states=text_features,  # [B, 512, dim] - key/value for cross-attention
            encoder_attention_mask=text_attention_mask,  # [B, 512] - will be extended by BertModel
            mode='multi_modal',
            output_hidden_states=True,  # Get all intermediate hidden states
            return_dict=True
        )
        
        # Get all hidden states (including input and all layer outputs)
        all_hidden_states = multimodal_output.hidden_states  # Tuple of [B, num_latents, dim] for each layer
        latents = multimodal_output.last_hidden_state  # Final output [B, num_latents, dim]
        
        # Compute L1 loss between latents before and after multimodal encoder
        l1_latent_loss = torch.mean(torch.abs(latents - latents_before_multimodal))
        
        
        # Decode: query points attend to latent code
        B, M, _ = query_points.shape
        
        if self.use_cnn:
            # Use CNN features for query points
            # Extract CNN feature pyramid from volume
            feature_pyramid = self.volume_encoder(modified_resized)  # List of [B, C, D, H, W]
            
            # Extract CNN features from multiple pyramid levels and concatenate
            # Checkpoint uses: concatenate last 3 levels [32, 64, 128] = 224 channels total
            # feature_pyramid structure: [stem(32), block0(32), block1(64), block2(128)]
            # We use levels 1, 2, 3 (indices 1, 2, 3) which are [32, 64, 128]
            # For 3D grid_sample, grid must be [B, D_out, H_out, W_out, 3]
            # We want to sample at M points, so reshape to [B, 1, 1, M, 3]
            query_coords_grid = query_coords_resized.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M, 3]
            
            # Extract features from last 3 pyramid levels
            feature_levels = feature_pyramid[1:4]  # [32, 64, 128] channels
            cnn_features_list = []
            for feat_map in feature_levels:
                # Extract features using grid_sample
                # Input: [B, C, D, H, W], Grid: [B, 1, 1, M, 3] -> Output: [B, C, 1, 1, M]
                feat = F.grid_sample(
                    feat_map,
                    query_coords_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )  # [B, C, 1, 1, M]
                feat = feat.squeeze(2).squeeze(2).transpose(1, 2)  # [B, M, C]
                cnn_features_list.append(feat)
            
            # Concatenate features from all levels: [B, M, 32+64+128] = [B, M, 224]
            cnn_features = torch.cat(cnn_features_list, dim=-1)  # [B, M, 224]
            
            # Get normalized query coordinates (for point encoding, aspect-ratio preserving)
            # query_points are already normalized in the same way as input_points
            query_coords_norm = query_points[..., :3]  # [B, M, 3]
            
            # Get query labels from input volume (if provided)
            B, M = query_coords_norm.shape[:2]
            if query_labels_input is not None:
                # query_labels_input is [B, M] or [M], reshape to [B, M, 1]
                if query_labels_input.dim() == 1:
                    query_labels_input = query_labels_input.unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
                elif query_labels_input.dim() == 2:
                    query_labels_input = query_labels_input.unsqueeze(-1)  # [B, M, 1]
                else:
                    query_labels_input = query_labels_input  # Already [B, M, 1]
            else:
                # Create zeros if not provided (for backward compatibility)
                query_labels_input = torch.zeros(B, M, 1, device=cnn_features.device, dtype=cnn_features.dtype)
            
            # Concatenate CNN features with normalized coordinates and input volume labels
            cnn_coords_features = torch.cat([cnn_features, query_coords_norm, query_labels_input], dim=-1)  # [B, M, 224+3+1] = [B, M, 228]
            
            # Process through MLP to get query embeddings
            query_embeddings = self.query_point_encoding_mlp(cnn_coords_features)  # [B, M, dim]
        else:
            # No CNN: use only query coordinates through MLP
            # Get normalized query coordinates (for point encoding, aspect-ratio preserving)
            query_coords_norm = query_points[..., :3]  # [B, M, 3]
            
            # Process through MLP to get query embeddings
            query_embeddings = self.query_point_encoding_mlp(query_coords_norm)  # [B, M, dim]
        
        # Query cross attention
        query_features = self.query_cross_attend(query_embeddings, context=latents)
        # Apply feedforward layers with residual connections
        if exists(self.query_ff_layers):
            for i, query_ff in enumerate(self.query_ff_layers):
                tmp = query_ff(query_features)
                query_features = query_features + tmp
        # Predict labels for query points
        query_logits = self.to_logits(query_features)  # [B, M, num_classes]
        
        result = {
            "logits": query_logits,
            "l1_latent_loss": l1_latent_loss,  # L1 regularization between latents before and after multimodal encoder
        }
      
        return result
    
    def extract_shared_features(self, input_points: torch.Tensor, texts,
                                modified_resized: torch.Tensor = None, class_hint=None):
        """
        Extract shared features that don't depend on query points.
        This includes multimodal latents and CNN feature pyramid.

        ``class_hint`` is accepted so existing inference callers do not fail.
        The current network does not use it.
        
        Args:
            input_points: [B, N, 4] input point cloud
            texts: Text instructions
            modified_resized: [B, 1, 128, 128, 128] shape only, or [B, 2, 128, 128, 128] with the scan
            
        Returns:
            Dictionary with:
                - latents: [B, num_latents, dim] - multimodal latent code
                - feature_pyramid: List of feature maps from CNN encoder
              
        """
        del class_hint
        B = input_points.shape[0]
        
        # Encode input points to latent code
        input_embeddings = self.point_embed(input_points)  # [B, N, dim]
        
        # Initialize latents from tokens
        latents = repeat(self.latent_tokens, "n d -> b n d", b=B)  # [B, num_latents, dim]
        
        # Cross-attention: latents attend to input points
        cross_attn, cross_ff = self.cross_attend
        latents_immediate = cross_attn(latents, context=input_embeddings) + latents
        latents = cross_ff(latents_immediate) + latents_immediate
        
        # Store latents before multimodal encoder for L1 regularization
        latents_before_multimodal = latents.clone()
        
        # Multimodal fusion
        text_features, text_attention_mask = self.text_encoder(texts, batch_size=B)  # [B, 512, dim], [B, 512]
        latents_attention_mask = torch.ones(B, latents.size(1), dtype=torch.long, device=latents.device)
        
        multimodal_output = self.multimodal_encoder(
            encoder_embeds=latents,
            attention_mask=latents_attention_mask,
            encoder_hidden_states=text_features,
            encoder_attention_mask=text_attention_mask,
            mode='multi_modal',
            output_hidden_states=True,
            return_dict=True
        )
        
        all_hidden_states = multimodal_output.hidden_states
        latents = multimodal_output.last_hidden_state  # [B, num_latents, dim]
        
        # Compute L1 loss between latents before and after multimodal encoder
        l1_latent_loss = torch.mean(torch.abs(latents - latents_before_multimodal))
        
        
        # Extract CNN feature pyramid from volume (only if CNN is used)
        if self.use_cnn:
            feature_pyramid = self.volume_encoder(modified_resized)  # List of [B, C, D, H, W]
        else:
            feature_pyramid = None
        
        result = {
            "latents": latents,
            "feature_pyramid": feature_pyramid,
            "l1_latent_loss": l1_latent_loss,  # L1 regularization between latents before and after multimodal encoder
        }
        return result
    
    def forward_query_points(self, query_points: torch.Tensor, query_coords_resized: torch.Tensor,
                            latents: torch.Tensor, feature_pyramid: list = None, query_labels_input: torch.Tensor = None):
        """
        Forward pass for query points using pre-computed shared features.
        
        Args:
            query_points: [B, M, 3] query point coordinates (normalized, aspect-ratio preserving)
            query_coords_resized: [B, M, 3] query coordinates for grid sampling (normalized to [-1, 1] per dimension)
            latents: [B, num_latents, dim] pre-computed multimodal latents
            feature_pyramid: List of feature maps from CNN encoder (only used if use_cnn=True)
            
        Returns:
            Dictionary with:
                - logits: [B, M, num_classes] prediction logits
        """
        B, M, _ = query_points.shape
        
        if self.use_cnn:
            # Extract CNN features from multiple pyramid levels and concatenate
            # Checkpoint uses: concatenate last 3 levels [32, 64, 128] = 224 channels total
            # feature_pyramid structure: [stem(32), block0(32), block1(64), block2(128)]
            # We use levels 1, 2, 3 (indices 1, 2, 3) which are [32, 64, 128]
            # For 3D grid_sample, grid must be [B, D_out, H_out, W_out, 3]
            # We want to sample at M points, so reshape to [B, 1, 1, M, 3]
            query_coords_grid = query_coords_resized.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M, 3]
            
            # Extract features from last 3 pyramid levels
            feature_levels = feature_pyramid[1:4]  # [32, 64, 128] channels
            cnn_features_list = []
            for feat_map in feature_levels:
                # Extract features using grid_sample
                # Input: [B, C, D, H, W], Grid: [B, 1, 1, M, 3] -> Output: [B, C, 1, 1, M]
                feat = F.grid_sample(
                    feat_map,
                    query_coords_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )  # [B, C, 1, 1, M]
                feat = feat.squeeze(2).squeeze(2).transpose(1, 2)  # [B, M, C]
                cnn_features_list.append(feat)
            
            # Concatenate features from all levels: [B, M, 32+64+128] = [B, M, 224]
            cnn_features = torch.cat(cnn_features_list, dim=-1)  # [B, M, 224]
            
            # Get normalized query coordinates (for point encoding, aspect-ratio preserving)
            query_coords_norm = query_points[..., :3]  # [B, M, 3]
            
            # Get query labels from input volume (if provided)
            B, M = query_coords_norm.shape[:2]
            if query_labels_input is not None:
                # query_labels_input is [B, M] or [M], reshape to [B, M, 1]
                if query_labels_input.dim() == 1:
                    query_labels_input = query_labels_input.unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
                elif query_labels_input.dim() == 2:
                    query_labels_input = query_labels_input.unsqueeze(-1)  # [B, M, 1]
                else:
                    query_labels_input = query_labels_input  # Already [B, M, 1]
            else:
                # Create zeros if not provided (for backward compatibility)
                query_labels_input = torch.zeros(B, M, 1, device=cnn_features.device, dtype=cnn_features.dtype)
            
            # Concatenate CNN features with normalized coordinates and input volume labels
            cnn_coords_features = torch.cat([cnn_features, query_coords_norm, query_labels_input], dim=-1)  # [B, M, 224+3+1] = [B, M, 228]
            
            # Process through MLP to get query embeddings
            query_embeddings = self.query_point_encoding_mlp(cnn_coords_features)  # [B, M, dim]
        else:
            # No CNN: use only query coordinates through MLP
            # Get normalized query coordinates (for point encoding, aspect-ratio preserving)
            query_coords_norm = query_points[..., :3]  # [B, M, 3]
            
            # Process through MLP to get query embeddings (no self-attention)
            query_embeddings = self.query_point_encoding_mlp(query_coords_norm)  # [B, M, dim]
        
        # Cross-attention
        query_features = self.query_cross_attend(query_embeddings, context=latents)
        
        # Feedforward layers
        if self.query_ff_layers is not None:
            for query_ff in self.query_ff_layers:
                tmp = query_ff(query_features)
                query_features = query_features + tmp
        
        # Predictions
        query_logits = self.to_logits(query_features)  # [B, M, num_classes]
        
        return {
            "logits": query_logits,
        }


def build_point_transformer(
    *,
    input_points: int,
    query_points: int,
    point_feature_dim: int,
    num_classes: int = 14,
    num_latents: int = 256,
    depth: int = 8,
    dim: int = 256,
    heads: int = 8,
    dim_head: int = 64,
    decoder_ff: bool = True,
    use_cnn: bool = True,
    use_image: bool = False,
):
    return PointTransformer(
        num_points=input_points,
        num_query_points=query_points,
        point_feature_dim=point_feature_dim,
        num_classes=num_classes,
        num_latents=num_latents,
        depth=depth,
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        decoder_ff=decoder_ff,
        use_cnn=use_cnn,
        use_image=use_image,
    )
