import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import utils
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from text import cleaned_text_to_sequence


def _load_semantic_pt(path: str, device: torch.device, is_half: bool) -> torch.Tensor:
    obj: Any = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "last_hidden_state" in obj:
        obj = obj["last_hidden_state"]
    if not isinstance(obj, torch.Tensor):
        obj = torch.tensor(obj)
    x = obj
    # accept [T, C], [B, T, C], [B, C, T]
    if x.dim() == 2:
        pass
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(f"semantic pt must be 2D/3D tensor, got shape={tuple(x.shape)} from {path}")

    x = x.to(device)
    if is_half:
        x = x.half()
    return x


def _semantic_to_bct(semantic: torch.Tensor, semantic_dim: int) -> torch.Tensor:
    """
    Normalize semantic tensor to shape [B, C, T] (float/half preserved).
    Accepts:
    - [T, C]
    - [B, T, C]
    - [B, C, T]
    """
    if semantic.dim() == 2:
        # [T, C] -> [1, C, T]
        semantic = semantic.unsqueeze(0).transpose(1, 2)
    elif semantic.dim() == 3:
        # If see [B, T, C] -> transpose
        if semantic.shape[1] != semantic_dim and semantic.shape[2] == semantic_dim:
            semantic = semantic.transpose(1, 2)
    else:
        raise ValueError(f"semantic must be 2D/3D tensor, got shape={tuple(semantic.shape)}")

    if semantic.shape[1] != semantic_dim:
        raise ValueError(
            f"semantic channel mismatch: expected C={semantic_dim}, got {semantic.shape[1]} (shape={tuple(semantic.shape)})"
        )
    return semantic


def _match_semantic_len(semantic_bct: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Match semantic length to target_len on time axis.
    Training pipeline interpolates / pads semantic to spectrogram length, so do the same for inference.
    """
    cur = int(semantic_bct.shape[-1])
    target_len = int(target_len)
    if cur == target_len:
        return semantic_bct
    if abs(cur - target_len) <= 1:
        if cur < target_len:
            return F.pad(semantic_bct.float(), (0, target_len - cur), mode="replicate").to(semantic_bct.dtype)
        return semantic_bct[..., :target_len]
    return F.interpolate(semantic_bct.float(), size=target_len, mode="linear", align_corners=False).to(semantic_bct.dtype)


def _find_phoneme_tokens(name2text_path: str, utt: str) -> str:
    with open(name2text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            k, phonemes = parts[0], parts[1]
            if k == utt:
                return phonemes
    raise FileNotFoundError(f"utt not found in {name2text_path}: {utt}")


def _load_refer_spec(
    wav_path: str,
    device: torch.device,
    is_half: bool,
    sampling_rate: int,
    filter_length: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sampling_rate:
        # KISS: minimal resample; torchaudio may not be available here.
        import librosa

        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=sampling_rate)
    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    spec = spectrogram_torch(
        wav_t,
        filter_length,
        sampling_rate,
        hop_length,
        win_length,
        center=False,
    )
    if is_half:
        spec = spec.half()
    return spec


def main():
    ap = argparse.ArgumentParser(
        description="Infer/reconstruct waveform from continuous semantic (.pt) + phonemes + reference audio (audio_tower/omni training path)."
    )
    ap.add_argument("-c", "--config", required=True, help="Stage2 config json, e.g. GPT_SoVITS/configs/s2_omni2048.json")
    ap.add_argument("--sovits_ckpt", required=True, help="SoVITS generator checkpoint path, e.g. exp_dir/logs_s2_v2/G_5000.pth")

    ap.add_argument("--exp_dir", default=None, help="Experiment dir that contains 2-name2text.txt / 4-audio_tower / 5-wav32k")
    ap.add_argument("--utt", default=None, help="Utterance filename key, e.g. xxx.wav (must match 2-name2text.txt and 5-wav32k)")

    ap.add_argument("--semantic_pt", default=None, help="Path to semantic pt. If not set, use {exp_dir}/4-audio_tower/{utt}.pt")
    ap.add_argument("--ref_wav", default=None, help="Reference wav. If not set, use {exp_dir}/5-wav32k/{utt}")
    ap.add_argument("--phonemes", default=None, help="Space-separated phoneme tokens. If not set, read from {exp_dir}/2-name2text.txt")

    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true", help="Use fp16 (only recommended on CUDA).")
    ap.add_argument(
        "--no_match_len",
        action="store_true",
        help="Do NOT match semantic length to reference spectrogram length (not recommended).",
    )
    ap.add_argument("-o", "--out", required=True, help="Output wav path")

    args = ap.parse_args()

    hps = utils.get_hparams_from_file(args.config)

    device = torch.device(args.device)
    is_half = bool(args.half) and device.type == "cuda"

    exp_dir = args.exp_dir or getattr(hps.data, "exp_dir", None)
    if (args.utt is not None) and (exp_dir is None):
        raise ValueError("--exp_dir is required when using --utt")

    utt = args.utt
    semantic_pt = args.semantic_pt
    ref_wav = args.ref_wav
    phonemes = args.phonemes

    if utt is not None:
        if semantic_pt is None:
            semantic_pt = os.path.join(exp_dir, getattr(hps.data, "content_feature_dir", "4-audio_tower"), f"{utt}.pt")
        if ref_wav is None:
            ref_wav = os.path.join(exp_dir, "5-wav32k", utt)
        if phonemes is None:
            name2text = os.path.join(exp_dir, "2-name2text.txt")
            phonemes = _find_phoneme_tokens(name2text, utt)

    if semantic_pt is None or ref_wav is None or phonemes is None:
        raise ValueError("Need semantic_pt + ref_wav + phonemes (or provide --exp_dir + --utt to auto-locate them).")

    # Build model
    spec_channels = int(hps.data.filter_length // 2 + 1)
    segment_size = int(hps.train.segment_size // hps.data.hop_length)
    net_g = SynthesizerTrn(
        spec_channels,
        segment_size,
        n_speakers=int(hps.data.n_speakers),
        **hps.model,
    ).to(device)
    net_g.eval()
    if is_half:
        net_g = net_g.half()

    # Load checkpoint (training-style checkpoint with key "model")
    utils.load_checkpoint(args.sovits_ckpt, net_g, optimizer=None, skip_optimizer=True)

    # Prepare inputs
    semantic = _load_semantic_pt(semantic_pt, device=device, is_half=is_half)

    phone_tokens = [p for p in phonemes.strip().split(" ") if p]
    phone_ids = cleaned_text_to_sequence(phone_tokens, getattr(hps.model, "version", "v2"))
    phones = torch.LongTensor(phone_ids).unsqueeze(0).to(device)

    refer_spec = _load_refer_spec(
        ref_wav,
        device=device,
        is_half=is_half,
        sampling_rate=int(hps.data.sampling_rate),
        filter_length=int(hps.data.filter_length),
        hop_length=int(hps.data.hop_length),
        win_length=int(hps.data.win_length),
    )

    # IMPORTANT: match semantic length to spectrogram frames (same as training pipeline).
    semantic_bct = _semantic_to_bct(semantic, semantic_dim=int(hps.model.semantic_dim))
    if not args.no_match_len:
        semantic_bct = _match_semantic_len(semantic_bct, target_len=int(refer_spec.shape[-1]))

    with torch.no_grad():
        audio = net_g.decode_from_semantic(
            semantic=semantic_bct,
            text=phones,
            refer=refer_spec,
            noise_scale=float(args.noise_scale),
            speed=float(args.speed),
        )
    audio = audio.squeeze().detach().cpu().float().numpy()
    sf.write(args.out, audio, int(hps.data.sampling_rate))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

