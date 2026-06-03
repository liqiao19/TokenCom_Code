# Pretrained Checkpoints

Large pretrained checkpoint files are not tracked in Git.

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

`VQGAN/model.yaml` is tracked because it is small and required for model construction. The `.pth` and `.ckpt` checkpoint files should be downloaded separately or hosted through a model/data platform such as Hugging Face Hub, Zenodo, or GitHub Releases with Git LFS.

This project acknowledges the open-source MaskGIT resources from Halton-MaskGIT: https://github.com/valeoai/Halton-MaskGIT/tree/v1.0
