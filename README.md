# Adaptive Feature Blending (AFB)

Official PyTorch implementation of the BMVC 2025 paper

**Learning Event-guided Exposure-agnostic Video Frame Interpolation via Adaptive Feature Blending**<br>
Junsik Jung, Yoonki Cho, Woo Jae Kim, Lin Wang, Sung-eui Yoon<br>
[[arXiv]](https://arxiv.org/abs/2510.22565)

## Overview

Given two consecutive blurry frames (I0, I1) captured under unknown exposure times, the events stacked over
both shutter periods E_N, and a target timestamp τ, AFB restores the sharp frame at τ with two modules:

- **Target-adaptive Event Sampling (TES)** samples events around τ and the unknown exposure of each frame
  ([`models/tes.py`](models/tes.py)).
- **Target-adaptive Importance Mapping (TIM)** predicts an importance map ω_τ from the temporal proximity and
  spatial relevance of the features, which adaptively blends them as F_τ = ω_τ ⊗ F0 ⊕ (1 − ω_τ) ⊗ F1
  ([`models/tim.py`](models/tim.py)).

```
AFB
├── models/
│   ├── afb.py          # full network (Fig. 2)
│   ├── tes.py          # TES module (Sec. 3.2, Fig. 3(a))
│   ├── tim.py          # TIM module (Sec. 3.3, Fig. 3(b))
│   ├── modules.py      # encoders, cross-modal fusion, decoder, positional encoding
│   └── refine.py       # refinement network
├── data/               # GoPro dataset and event stacking
├── utils/metrics.py    # PSNR / SSIM / LPIPS
├── test.py             # evaluation
└── scripts/test_gopro.sh
```

## Installation

```bash
conda create -n afb python=3.9
conda activate afb
pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Data

We use GoPro at 180×320 with synthetic events from ESIM. Consecutive shutter periods are packed into `.npz` files:

```
<data_dir>/test/<video>/<xxxxxx_yyyyyy>.npz
    left_frames, right_frames   dict {i: (H, W, 3) uint8 BGR sharp frame}, one entry per frame of the shutter period
    left_event,  right_event    dict {'x', 'y', 't', 'p'} events recorded during each shutter period
```

A shutter period contains 10 frames for **GoPro-10⇓** and 16 frames for **GoPro-16⇓**. The blurry inputs are
synthesized on the fly by averaging the first m sharp frames of each period.

## Pretrained Models

| Setting    | Download                                                                                          | `--num_frames` |
|------------|---------------------------------------------------------------------------------------------------|----------------|
| GoPro-10⇓  | [afb_gopro_10down.pth](https://github.com/Juns1105/AFB/releases/download/v1.0/afb_gopro_10down.pth) | 20             |
| GoPro-16⇓  | [afb_gopro_16down.pth](https://github.com/Juns1105/AFB/releases/download/v1.0/afb_gopro_16down.pth) | 32             |

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/Juns1105/AFB/releases/download/v1.0/afb_gopro_10down.pth
wget -P checkpoints https://github.com/Juns1105/AFB/releases/download/v1.0/afb_gopro_16down.pth
```

## Evaluation

```bash
# GoPro-10⇓, symmetric exposure with m = 5 (5+5)
python test.py --data_dir <GoPro-10⇓ root> --checkpoint checkpoints/afb_gopro_10down.pth \
    --num_frames 20 --left_m 5 --right_m 5

# GoPro-16⇓, RandEx (m sampled randomly for each input)
python test.py --data_dir <GoPro-16⇓ root> --checkpoint checkpoints/afb_gopro_16down.pth \
    --num_frames 32 --left_m 0 --right_m 0
```

`scripts/test_gopro.sh` runs all protocols of the paper (m ∈ {1, 5, 9} for 10⇓, m ∈ {1, 5, 11, 15} for 16⇓, and RandEx).
PSNR, SSIM and LPIPS are averaged over all restored frames. Use `--save_dir` to write the metrics and
`--save_images` to also save the restored frames.

## Results

| GoPro-10⇓ | 9+1 | 5+5 | 1+9 | RandEx |
|---|---|---|---|---|
| PSNR / SSIM / LPIPS | 33.22 / 0.960 / 0.050 | 33.61 / 0.963 / 0.042 | 32.87 / 0.954 / 0.048 | 33.39 / 0.961 / 0.045 |

| GoPro-16⇓ | 15+1 | 11+5 | 5+11 | 1+15 | RandEx |
|---|---|---|---|---|---|
| PSNR / SSIM / LPIPS | 30.97 / 0.937 / 0.081 | 31.39 / 0.941 / 0.074 | 31.18 / 0.938 / 0.075 | 30.07 / 0.925 / 0.085 | 31.14 / 0.938 / 0.076 |

## Citation

```bibtex
@inproceedings{jung2025afb,
  title     = {Learning Event-guided Exposure-agnostic Video Frame Interpolation via Adaptive Feature Blending},
  author    = {Jung, Junsik and Cho, Yoonki and Kim, Woo Jae and Wang, Lin and Yoon, Sung-eui},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2025}
}
```

## Acknowledgements

Our network is built upon [EFNet](https://github.com/AHupuJR/EFNet). We thank the authors for sharing their code.
The refinement network is adapted from [EVDI](https://github.com/XiangZ-0/EVDI), and the event stacking from
[event_utils](https://github.com/TimoStoff/event_utils).

## License

This project is released under the [MIT License](LICENSE).
