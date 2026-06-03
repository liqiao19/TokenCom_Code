# Pretrained Checkpoints

Large pretrained checkpoint files are not tracked in Git.

Download the pretrained weights from Hugging Face: [llvictorll/Maskgit-pytorch](https://huggingface.co/llvictorll/Maskgit-pytorch/tree/main).

Expected local layout:

```text
pretrained_maskgit/
  MaskGIT/
    MaskGIT_ImageNet_256.pth
    MaskGIT_ImageNet_512.pth
  VQGAN/
    model.yaml
    last.ckpt
```

`VQGAN/model.yaml` is tracked because it is small and required for model construction. The `.pth` and `.ckpt` checkpoint files should be downloaded separately and placed in the paths shown above.

This project acknowledges the open-source MaskGIT resources from [Halton-MaskGIT](https://github.com/valeoai/Halton-MaskGIT/tree/v1.0).
