# CowTalk model

The network edits an erroneous segmentation from a text instruction. It reads a point cloud, not the raw label volume. `README.md` describes how those point clouds are built.

## Inputs

- Input points `[B, N, 4]`: normalized coordinates plus the current label at each point.
- Query points `[B, M, 3]`: coordinates to label. Their supervision is the segmentation after the instructed edits.
- Text: one instruction string per batch item.
- Resized volume for the CNN. Shape only: `[B, 1, 128, 128, 128]`, the multi-class error labels. With `--use-image`: `[B, 2, 128, 128, 128]`, those labels plus the aligned scan. The pyramid widths stay the same; only the first convolution changes.

## Forward pass

1. **Point embedding.** A two-layer MLP maps each input point to `dim` (512). Position embeddings are off.
2. **Shape latents.** 512 learnable latent tokens cross-attend to the input points, then pass through a feed-forward block.
3. **Text encoder.** A frozen `bert-base-uncased`, loaded into the BERT in `albef/xbert.py`, embeds up to 512 tokens. A linear layer maps 768 dimensions to 512. This encoder is not trained.
4. **Text fusion.** A second BERT from `albef/xbert.py`, with `fusion_layer=0` and depth 6, takes the shape latents as queries and the text tokens as keys and values. Every layer can attend to the instruction. An L1 penalty keeps the fused latents close to the latents from step 2.
5. **Query features.** If the CNN is enabled, three pyramid levels (32, 64, and 128 channels) are sampled at each query point and concatenated with the query coordinate and the input label (228 values), then mapped to 512. Without the CNN, only the query coordinate goes through that MLP.
6. **Prediction.** Query features cross-attend to the fused latents, pass through two feed-forward blocks, and a linear layer predicts one of 14 labels (background plus classes 1–13).

The CNN stem uses channel widths `[32, 32, 64, 128]`. The first convolution has one input channel for the shape-only experiment and two when the scan is included.

## What this network does not include

Class-hint latents are not part of the forward pass. Inference scripts may still pass a `class_hint` tensor; `extract_shared_features` accepts it and does not use it. The instruction text is the only language input that changes the latents.
