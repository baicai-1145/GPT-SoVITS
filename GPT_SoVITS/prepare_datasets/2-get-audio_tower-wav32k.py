# -*- coding: utf-8 -*-
"""
用 Qwen3-Omni audio_tower 提取连续隐空间特征（默认 2048-d），写入 exp_dir/4-audio_tower，并同时生成 32k wav。

与原 `2-get-hubert-wav32k.py` 保持相同的环境变量接口，便于在现有数据预处理流水线中替换：
- inp_text: 训练集清单（每行：wav|spk|lang|text）
- inp_wav_dir: 可选，若不为空则从该目录拼接 wav_name
- opt_dir: 实验目录（输出到 opt_dir/4-audio_tower 与 opt_dir/5-wav32k）
- i_part / all_parts: 分片处理
- is_half: 是否半精度（仅在 CUDA 上有效）
- audio_tower_dir: 默认 GPT_SoVITS/pretrained_models/audio_tower
- content_feature_dir: 可选，默认 `4-audio_tower`（特征输出目录名）

输出的 pt 文件为 dict 格式：
  {
    "last_hidden_state": Tensor[T, C],
    "meta": {...},
    "audio_encoder_config": {...}
  }

SoVITS 侧已做兼容，可直接读取 dict 并对齐长度。
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import sys
import traceback
from time import time as ttime
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
opt_dir = os.environ.get("opt_dir")
i_part = os.environ.get("i_part", "0")
all_parts = os.environ.get("all_parts", "1")

audio_tower_dir = os.environ.get(
    "audio_tower_dir", os.path.join("GPT_SoVITS", "pretrained_models", "audio_tower")
)

def clean_path(path: str) -> str:
    # Minimal replacement to avoid pulling heavy deps (e.g. gradio) from tools/my_utils.py.
    return path.strip().strip('"').strip("'")


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """
    Minimal WAV loader + resampler.
    Returns mono float32 in [-1, 1].
    """
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    # int16/int32/float -> float32 [-1,1]
    if np.issubdtype(data.dtype, np.integer):
        maxv = float(np.iinfo(data.dtype).max)
        wav = data.astype(np.float32) / maxv
    else:
        wav = data.astype(np.float32)
        # Some datasets store float wavs not normalized to [-1, 1]. Normalize defensively.
        absmax = float(np.max(np.abs(wav))) if wav.size else 0.0
        if absmax > 1.5 and np.isfinite(absmax):
            wav = (wav / absmax).astype(np.float32, copy=False)
    if int(sr) != int(target_sr):
        g = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        wav = resample_poly(wav, up, down).astype(np.float32, copy=False)
    return wav


def _pick_one(patterns: List[str], base_dir: str) -> str:
    for pat in patterns:
        cands = sorted(glob.glob(os.path.join(base_dir, pat)))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"找不到文件：base_dir={base_dir}, patterns={patterns}")


def my_save(obj, path: str):
    """fix issue: torch.save doesn't support chinese path"""
    dir_ = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(obj, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir_, name))


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    # 同 tools/extract_qwen3_omni_audio_tower_last_hidden_state.py
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def _load_audio_tower(audio_tower_dir: str, device: str, dtype: torch.dtype):
    cfg_path = _pick_one(["*audio_tower_config.json", "audio_tower_config.json"], audio_tower_dir)
    w_path = _pick_one(["*audio_tower.safetensors", "audio_tower.safetensors"], audio_tower_dir)

    import json

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**cfg_dict)
    model = Qwen3OmniMoeAudioEncoder(audio_cfg)

    from safetensors.torch import load_file

    sd = load_file(w_path)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"audio_tower load_state_dict mismatch. missing={missing}, unexpected={unexpected}")

    model.eval().to(device=device, dtype=dtype)
    return model, audio_cfg


def _make_feature_extractor(num_mel_bins: int):
    # 不依赖 preprocessor_config.json，直接按 WhisperFeatureExtractor 规则生成 log-mel（但 mel bins 取自 audio_cfg）
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    return WhisperFeatureExtractor(feature_size=int(num_mel_bins), sampling_rate=16000, return_attention_mask=True)


def main():
    assert inp_text, "缺少环境变量 inp_text"
    assert opt_dir, "缺少环境变量 opt_dir"

    content_feature_dir = os.environ.get("content_feature_dir", "4-audio_tower")
    hubert_dir = f"{opt_dir}/{content_feature_dir}"
    wav32dir = f"{opt_dir}/5-wav32k"
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        is_half = False

    dtype = torch.float16 if is_half else torch.float32

    audio_model, audio_cfg = _load_audio_tower(audio_tower_dir, device=device, dtype=dtype)
    fe = _make_feature_extractor(num_mel_bins=getattr(audio_cfg, "num_mel_bins", 128))

    maxx = 0.95
    alpha = 0.5

    def name2go(wav_name: str, wav_path: str):
        out_pt = f"{hubert_dir}/{wav_name}.pt"
        out_wav = f"{wav32dir}/{wav_name}"
        if os.path.exists(out_pt) and os.path.exists(out_wav):
            return

        # 1) wav32k：与原脚本一致的幅度策略
        tmp_audio32 = load_audio(wav_path, 32000)
        tmp_max = float(np.abs(tmp_audio32).max()) if tmp_audio32.size else 0.0
        if not np.isfinite(tmp_max) or tmp_max <= 1e-8:
            print("%s-filtered,too_silent_or_invalid" % wav_name)
            return
        if tmp_max > 2.2:
            print("%s-filtered,%s" % (wav_name, tmp_max))
            return
        tmp_audio32_i16 = (tmp_audio32 / tmp_max * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * tmp_audio32
        wavfile.write(out_wav, 32000, tmp_audio32_i16.astype("int16"))

        # 2) audio_tower 输入：16k mono float32
        wav16 = load_audio(wav_path, 16000).astype("float32", copy=False)
        if wav16.size < 400:
            # WhisperFeatureExtractor uses n_fft=400; too-short waveforms will crash torch.stft padding.
            print("%s-filtered,too_short_%d" % (wav_name, int(wav16.size)))
            return

        try:
            inputs = fe([wav16], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        except Exception as e:
            print("%s-filtered,feature_extractor_error:%s" % (wav_name, str(e)))
            return
        input_features = inputs["input_features"]  # [B, F, T]
        feature_attention_mask = inputs.get("attention_mask", None)  # [B, T]
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(input_features.shape[0], input_features.shape[-1], dtype=torch.long)

        feature_lens = feature_attention_mask.sum(-1).to(torch.long)  # [B]
        packed = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0).contiguous()  # [F, sum_T]

        packed = packed.to(device=device, dtype=dtype)
        feature_lens_dev = feature_lens.to(device=device)

        with torch.no_grad():
            try:
                out = audio_model(packed, feature_lens=feature_lens_dev, return_dict=True)
                last_hidden = out.last_hidden_state  # [sum_out_T, C]
            except Exception as e:
                print("%s-filtered,audio_tower_error:%s" % (wav_name, str(e)))
                return

        out_lens = _get_feat_extract_output_lengths(feature_lens).tolist()
        emb = torch.split(last_hidden, out_lens, dim=0)[0]  # [T_out, C]

        my_save(
            {
                "last_hidden_state": emb.detach().cpu(),
                "meta": {
                    "audio_path": wav_path,
                    "sampling_rate": 16000,
                    "feature_len": int(feature_lens.item()),
                    "output_len": int(emb.shape[0]),
                    "output_dim": int(emb.shape[1]),
                    "dtype": str(emb.dtype).replace("torch.", ""),
                    "audio_tower_dir": os.path.abspath(audio_tower_dir),
                },
                "audio_encoder_config": audio_cfg.to_dict(),
            },
            out_pt,
        )

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            # text can contain '|', so only split the first 3 separators.
            wav_name, _spk_name, _language, _text = line.split("|", 3)
            wav_name = clean_path(wav_name)
            if inp_wav_dir:
                wav_name = os.path.basename(wav_name)
                wav_path = f"{inp_wav_dir}/{wav_name}"
            else:
                wav_path = wav_name
                wav_name = os.path.basename(wav_name)
            name2go(wav_name, wav_path)
        except Exception:
            print(line, traceback.format_exc())


if __name__ == "__main__":
    main()

