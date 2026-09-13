"""Convert a checkpoint of the original research code to the module names used in this repository.

    python tools/convert_legacy_checkpoint.py --src epoch_0300.pth --dst checkpoints/afb_gopro_10down.pth --num_frames 20
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AFB  # noqa: E402

# (legacy pattern, new name); the first matching rule is applied.
RULES = [
    (r'^conv_01\.', 'frame_stem.'),
    (r'^conv_ev1\.', 'event_stem.'),
    (r'^ev_proc\.conv_ev1\.', 'tes.conv_event.'),
    (r'^ev_proc\.conv1_left\.', 'tes.conv_frame0.'),
    (r'^ev_proc\.conv1_right\.', 'tes.conv_frame1.'),
    (r'^ev_proc\.norm_ev\.', 'tes.norm_event.'),
    (r'^ev_proc\.norm_left\.', 'tes.norm_frame0.'),
    (r'^ev_proc\.norm_right\.', 'tes.norm_frame1.'),
    (r'^ev_proc\.t_proj\.', 'tes.t_proj.'),
    (r'^ev_proc\.conv_ev_left_last\.', 'tes.out_conv0.'),
    (r'^ev_proc\.conv_ev_right_last\.', 'tes.out_conv1.'),
    (r'^down_path_1\.(\d+)\.image_event_transformer\.norm1_image\.', r'frame_encoder.\1.fusion.norm_frame.'),
    (r'^down_path_1\.(\d+)\.image_event_transformer\.norm1_event\.', r'frame_encoder.\1.fusion.norm_event.'),
    (r'^down_path_1\.(\d+)\.image_event_transformer\.norm2\.', r'frame_encoder.\1.fusion.norm_ffn.'),
    (r'^down_path_1\.(\d+)\.image_event_transformer\.', r'frame_encoder.\1.fusion.'),
    (r'^down_path_1\.', 'frame_encoder.'),
    (r'^down_path_ev1\.(\d+)\.conv_before_merge\.', r'event_encoder.\1.out_conv.'),
    (r'^down_path_ev1\.', 'event_encoder.'),
    (r'^mask_gen\.(\d+)\.measure_sim\.norm1_event1\.', r'tim.\1.attention.norm_query.'),
    (r'^mask_gen\.(\d+)\.measure_sim\.norm1_event2\.', r'tim.\1.attention.norm_kv.'),
    (r'^mask_gen\.(\d+)\.measure_sim\.', r'tim.\1.attention.'),
    (r'^mask_gen\.(\d+)\.gen_conv_(\d)\.', r'tim.\1.head_conv_\2.'),
    (r'^mask_gen\.', 'tim.'),
    (r'^up_path_1\.(\d+)\.conv_block\.', r'decoder.\1.'),
    (r'^up_path_1\.', 'decoder.'),
    (r'^skip_conv_1\.', 'skip_convs.'),
    (r'^last_1\.', 'out_conv.'),
    (r'^fusion_net\.convBlock1\.', 'refine.in_conv.'),
    (r'^fusion_net\.Pre\.', 'refine.pre.'),
    (r'^fusion_net\.Post\.', 'refine.post.'),
    (r'^fusion_net\.resBlock(\d)\.conv_block\.0\.', r'refine.res\1.conv1.'),
    (r'^fusion_net\.resBlock(\d)\.conv_block\.3\.', r'refine.res\1.conv2.'),
    (r'^fusion_net\.conv\.', 'refine.out_conv.'),
    (r'^fusion_net\.', 'refine.'),
]


def convert_key(key):
    if key.startswith('module.'):
        key = key[len('module.'):]
    for pattern, repl in RULES:
        if re.match(pattern, key):
            key = re.sub(pattern, repl, key)
            break
    # LayerNorm2d stores its parameters directly instead of under `.body`.
    return re.sub(r'\.body\.(weight|bias)$', r'.\1', key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--num_frames', type=int, required=True)
    args = parser.parse_args()

    legacy = torch.load(args.src, map_location='cpu')
    state_dict = {convert_key(k): v for k, v in legacy.items()}

    AFB(num_bins=args.num_frames).load_state_dict(state_dict, strict=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(state_dict, args.dst)
    print(f'Saved {len(state_dict)} tensors to {args.dst}')


if __name__ == '__main__':
    main()
