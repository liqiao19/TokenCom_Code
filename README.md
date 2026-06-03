# TokenCom Code

Official implementation code for the IEEE Wireless Communications paper:

**Token Communications: A Large Model-Driven Framework for Cross-Modal Context-Aware Semantic Communications**  
L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, *IEEE Wireless Communications*, vol. 32, no. 5, pp. 80-88, October 2025.  
DOI: [10.1109/MWC.001.2500084](https://doi.org/10.1109/MWC.001.2500084)

TokCom is a large-model-driven semantic communication framework that represents visual content as discrete tokens, transmits them over noisy wireless channels, and reconstructs the semantic content with generative token modeling.

## What Is Included

- `TokCom_Demo.ipynb`: End-to-end TokCom demonstration notebook.
- `Network/`: Masked token transformer and VQGAN-related network modules.
- `Trainer/`: MaskGIT trainer and model-loading utilities.
- `Performance_metrics.py`: PSNR, LPIPS, and CLIP-based evaluation utilities.
- `synset_words.txt`: ImageNet synset mapping used by the demo.
- `pretrained_maskgit/VQGAN/model.yaml`: VQGAN model configuration.

Large pretrained checkpoint files are intentionally **not tracked by Git**. See [Pretrained Checkpoints](#pretrained-checkpoints).

## Installation

Create a Python environment first. Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd TokenCom_Code
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For running the demo notebook:

```bash
python -m ipykernel install --user --name tokencom
jupyter notebook TokCom_Demo.ipynb
```

If your CUDA version is different, install the PyTorch build that matches your local environment from the official PyTorch installation selector.

## Pretrained Checkpoints

The demo expects pretrained MaskGIT and VQGAN checkpoints at these local paths:

```text
pretrained_maskgit/
  MaskGIT/
    MaskGIT_ImageNet_256.pth
    MaskGIT_ImageNet_512.pth   # optional, depending on the experiment
  VQGAN/
    model.yaml                 # tracked in this repository
    last.ckpt
```

The checkpoint files are very large, approximately 0.9-2.1 GB each, so they are excluded from this GitHub repository to avoid GitHub file-size and quota issues.

Download the pretrained weights from Hugging Face: [llvictorll/Maskgit-pytorch](https://huggingface.co/llvictorll/Maskgit-pytorch/tree/main).

After downloading the checkpoints, place them in the paths shown above. The demo notebook currently uses `MaskGIT_ImageNet_256.pth` by default.

## Data

The notebook uses ImageNet-style class folders and `synset_words.txt` for class-index mapping. Set the dataset root in `TokCom_Demo.ipynb`, or export an environment variable before launching Jupyter:

```bash
export IMAGENET_ROOT=/path/to/ImageNet
```

The code does not include ImageNet images or other datasets.

## Running The Demo

1. Install dependencies.
2. Place the pretrained checkpoints under `pretrained_maskgit/`.
3. Set `IMAGENET_ROOT` or update `Config.data_folder` in `TokCom_Demo.ipynb`.
4. Run the notebook cells from top to bottom.

The notebook evaluates reconstruction quality across SNR levels and reports metrics including PSNR, LPIPS, and CLIP similarity.

## Acknowledgements

This project acknowledges the open-source MaskGIT resources from [Halton-MaskGIT](https://github.com/valeoai/Halton-MaskGIT/tree/v1.0).

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{qiao2025tokencom,
  author={Qiao, Li and Mashhadi, Mahdi Boloursaz and Gao, Zhen and Tafazolli, Rahim and Bennis, Mehdi and Niyato, Dusit},
  journal={IEEE Wireless Communications}, 
  title={Token Communications: A Large Model-Driven Framework for Cross-Modal Context-Aware Semantic Communications}, 
  year={2025},
  volume={32},
  number={5},
  pages={80-88},
  keywords={Token networks;Large language models;Context awareness;Semantic communication;Transmitters;Spectral efficiency;Context modeling;Complexity theory;Transformers},
  doi={10.1109/MWC.001.2500084}}

```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
