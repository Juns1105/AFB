# Adaptive Feature Blending (AFB)

Official PyTorch implementation of the BMVC 2025 paper

**Learning Event-guided Exposure-agnostic Video Frame Interpolation via Adaptive Feature Blending**<br>
Junsik Jung, Yoonki Cho, Woo Jae Kim, Lin Wang, Sung-eui Yoon<br>
[[Project Page]](https://sgvr.kaist.ac.kr/~jsjung/AFB/index.html) [[arXiv]](https://arxiv.org/abs/2510.22565)

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
├── utils/              # Charbonnier loss, PSNR / SSIM / LPIPS
├── train.py            # training
├── test.py             # evaluation
└── scripts/            # training and evaluation scripts
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
<data_dir>/{train,test}/<video>/<xxxxxx_yyyyyy>.npz
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

## Training

```bash
# GoPro-10⇓
python train.py --data_dir <GoPro-10⇓ root> --num_frames 20 --exp_name afb_gopro_10down --batch_size 36 --multi_gpu

# GoPro-16⇓
python train.py --data_dir <GoPro-16⇓ root> --num_frames 32 --exp_name afb_gopro_16down --batch_size 36 --multi_gpu
```

The model is trained end-to-end for 300 epochs with AdamW (learning rate 5e-4, cosine annealing) and the
Charbonnier loss on 128×128 crops. For every input, m is sampled randomly within the shutter period and the
target timestamp randomly among all frames. After each epoch the model is validated at a fixed target
timestamp (`--val_target`), and the checkpoint is saved to `<save_dir>/<exp_name>/models/` whenever the
validation PSNR improves.

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
