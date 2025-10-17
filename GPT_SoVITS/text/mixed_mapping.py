from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from text.LangSegmenter import LangSegmenter

CleanTextFn = Callable[[str, str, str], Tuple[Sequence[int], Sequence[int] | None, str, Dict[str, Any]]]


def _collect_text_segments(raw_text: str, ui_language: str) -> Tuple[List[str], List[str]]:
    textlist: List[str] = []
    langlist: List[str] = []

    normalized_lang = ui_language.replace("all_", "")
    if ui_language == "all_zh":
        for tmp in LangSegmenter.getTexts(raw_text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif ui_language == "all_yue":
        for tmp in LangSegmenter.getTexts(raw_text, "zh"):
            seg_lang = "yue" if tmp["lang"] == "zh" else tmp["lang"]
            langlist.append(seg_lang)
            textlist.append(tmp["text"])
    elif ui_language == "all_ja":
        for tmp in LangSegmenter.getTexts(raw_text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif ui_language == "all_ko":
        for tmp in LangSegmenter.getTexts(raw_text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif ui_language == "en":
        langlist.append("en")
        textlist.append(raw_text)
    elif ui_language == "auto":
        for tmp in LangSegmenter.getTexts(raw_text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif ui_language == "auto_yue":
        for tmp in LangSegmenter.getTexts(raw_text):
            seg_lang = "yue" if tmp["lang"] == "zh" else tmp["lang"]
            langlist.append(seg_lang)
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(raw_text):
            if langlist:
                latest_lang = langlist[-1]
                if (tmp["lang"] == "en" and latest_lang == "en") or (tmp["lang"] != "en" and latest_lang != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(normalized_lang)
            textlist.append(tmp["text"])

    return textlist, langlist


def _tokenize_words(norm_text: str, punctuation: Iterable[str]) -> List[Tuple[str, int, int]]:
    punctuation_set = set(punctuation)
    return [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"\S+", norm_text)
        if not all((ch in punctuation_set) for ch in match.group())
    ]


def build_mixed_mappings(
    text: str,
    ui_language: str,
    version: str,
    clean_text_fn: CleanTextFn,
    punctuation: Iterable[str],
) -> Tuple[str, List[int], List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """
    将原始文本、归一化文本与音素对齐信息打包。该函数会复用语言分段逻辑，并确保
    段落、词级与音素级间的映射保持一致。
    """
    normalized_text = re.sub(r" {2,}", " ", text)
    text_segments, lang_segments = _collect_text_segments(normalized_text, ui_language)

    norm_text_parts: List[str] = []
    original_text_parts: List[str] = []
    ph_to_char: List[int] = []
    ph_to_word: List[int] = []
    word_norm_ranges: List[Tuple[int, int]] = []
    segments_global: List[Dict[str, Any]] = []
    segments_raw_global: List[Dict[str, Any]] = []
    segment_for_norm_index: List[int] = []

    norm_offset = 0
    orig_offset = 0

    punctuation_set = set(punctuation)

    for seg_text, seg_lang in zip(text_segments, lang_segments):
        seg_phones, seg_word2ph, seg_norm, seg_mapping = clean_text_fn(seg_text, seg_lang, version)
        seg_norm_len = len(seg_norm)
        seg_norm_start = norm_offset
        seg_orig_start = orig_offset

        norm_text_parts.append(seg_norm)
        original_text_parts.append(seg_text)

        char_spans_local = seg_mapping.get("char_spans", [])
        segments_local = seg_mapping.get("segments", [])
        segments_raw_local = seg_mapping.get("segments_raw", [])
        token_matches: List[Tuple[str, int, int]] = []

        if seg_lang in {"en", "ko"} and seg_word2ph:
            token_matches = _tokenize_words(seg_norm, punctuation_set)

        if not segments_local and char_spans_local:
            segments_local = []
            start_idx = None
            current_span = None
            for idx, span in enumerate(char_spans_local):
                if span == (-1, -1):
                    if start_idx is not None and current_span is not None:
                        segments_local.append({
                            "norm_start": start_idx,
                            "norm_end": idx,
                            "orig_start": current_span[0],
                            "orig_end": current_span[1],
                            "text": seg_norm[start_idx:idx],
                        })
                    start_idx = None
                    current_span = None
                    continue
                if start_idx is None or span != current_span:
                    if start_idx is not None and current_span is not None:
                        segments_local.append({
                            "norm_start": start_idx,
                            "norm_end": idx,
                            "orig_start": current_span[0],
                            "orig_end": current_span[1],
                            "text": seg_norm[start_idx:idx],
                        })
                    start_idx = idx
                    current_span = span
            if start_idx is not None and current_span is not None:
                segments_local.append({
                    "norm_start": start_idx,
                    "norm_end": len(char_spans_local),
                    "orig_start": current_span[0],
                    "orig_end": current_span[1],
                    "text": seg_norm[start_idx:],
                })

        if seg_lang in {"en", "ko"} and token_matches:
            word_segments = []
            for _, start_rel, end_rel in token_matches:
                norm_slice = seg_norm[start_rel:end_rel]
                orig_candidates = []
                for idx in range(start_rel, min(end_rel, len(char_spans_local))):
                    span = char_spans_local[idx]
                    if span and span != (-1, -1):
                        orig_candidates.append(span)
                if orig_candidates:
                    local_orig_start = min(span[0] for span in orig_candidates)
                    local_orig_end = max(span[1] for span in orig_candidates)
                else:
                    local_orig_start = -1
                    local_orig_end = -1
                word_segments.append({
                    "norm_start": start_rel,
                    "norm_end": end_rel,
                    "orig_start": local_orig_start,
                    "orig_end": local_orig_end,
                    "text": norm_slice,
                })
            merged_segments = []
            merged_by_key: Dict[int, Dict[str, Any]] = {}
            for seg in word_segments:
                key = seg["orig_start"]
                if key not in merged_by_key or key == -1:
                    merged_seg = {
                        "orig_start": seg["orig_start"],
                        "orig_end": seg["orig_end"],
                        "norm_start": seg["norm_start"],
                        "norm_end": seg["norm_end"],
                        "text": seg["text"],
                    }
                    merged_by_key[key] = merged_seg
                    merged_segments.append(merged_seg)
                else:
                    existing = merged_by_key[key]
                    existing["orig_end"] = max(existing.get("orig_end", -1), seg.get("orig_end", -1))
                    existing["norm_start"] = min(existing["norm_start"], seg["norm_start"])
                    existing["norm_end"] = max(existing["norm_end"], seg["norm_end"])
                    existing["text"] = seg_norm[existing["norm_start"]:existing["norm_end"]]
            merged_segments.sort(key=lambda item: item["norm_start"])
            segments_local = merged_segments
            segments_raw_local = [seg.copy() for seg in merged_segments]

        for seg_entry in segments_local:
            local_norm_start = seg_entry.get("norm_start", 0)
            local_norm_end = seg_entry.get("norm_end", 0)
            global_norm_start = seg_norm_start + local_norm_start
            global_norm_end = seg_norm_start + local_norm_end

            local_orig_start = seg_entry.get("orig_start", -1)
            local_orig_end = seg_entry.get("orig_end", -1)
            if (
                local_orig_start is not None
                and local_orig_start >= 0
                and local_orig_end is not None
                and local_orig_end > local_orig_start
            ):
                global_orig_start = seg_orig_start + local_orig_start
                global_orig_end = seg_orig_start + local_orig_end
                original_slice = seg_text[local_orig_start:local_orig_end]
            else:
                global_orig_start = -1
                global_orig_end = -1
                original_slice = seg_entry.get("text", seg_norm[local_norm_start:local_norm_end])

            text_norm_slice = seg_entry.get("text", seg_norm[local_norm_start:local_norm_end])

            seg_id = len(segments_global)
            segments_global.append({
                "norm_start": global_norm_start,
                "norm_end": global_norm_end,
                "orig_start": global_orig_start,
                "orig_end": global_orig_end,
                "text_original": original_slice,
                "text_norm": text_norm_slice,
                "text_norm_indices": (global_norm_start, global_norm_end),
                "language": seg_lang,
            })

            if len(segment_for_norm_index) < global_norm_end:
                segment_for_norm_index.extend([-1] * (global_norm_end - len(segment_for_norm_index)))
            for idx in range(global_norm_start, global_norm_end):
                segment_for_norm_index[idx] = seg_id

        if segments_raw_local:
            for seg_entry in segments_raw_local:
                local_norm_start = seg_entry.get("norm_start", 0)
                local_norm_end = seg_entry.get("norm_end", 0)
                global_norm_start = seg_norm_start + local_norm_start
                global_norm_end = seg_norm_start + local_norm_end

                local_orig_start = seg_entry.get("orig_start", -1)
                local_orig_end = seg_entry.get("orig_end", -1)
                if (
                    local_orig_start is not None
                    and local_orig_start >= 0
                    and local_orig_end is not None
                    and local_orig_end > local_orig_start
                ):
                    global_orig_start = seg_orig_start + local_orig_start
                    global_orig_end = seg_orig_start + local_orig_end
                    original_slice = seg_text[local_orig_start:local_orig_end]
                else:
                    global_orig_start = -1
                    global_orig_end = -1
                    original_slice = seg_entry.get("text", seg_norm[local_norm_start:local_norm_end])

                text_norm_slice = seg_entry.get("text", seg_norm[local_norm_start:local_norm_end])

                segments_raw_global.append({
                    "norm_start": global_norm_start,
                    "norm_end": global_norm_end,
                    "orig_start": global_orig_start,
                    "orig_end": global_orig_end,
                    "text_original": original_slice,
                    "text_norm": text_norm_slice,
                    "segment_index": len(segments_global) - 1 if segments_global else -1,
                })

        seg_word2ph = seg_word2ph or []
        if seg_lang in {"zh", "yue", "ja"} and seg_word2ph:
            char_base_idx = seg_norm_start
            for ch_idx, cnt in enumerate(seg_word2ph):
                global_char_idx = char_base_idx + ch_idx
                ph_to_char += [global_char_idx] * cnt
                ph_to_word += [len(word_norm_ranges)] * cnt
                word_norm_ranges.append((global_char_idx, global_char_idx + 1))
        elif seg_lang in {"en", "ko"} and seg_word2ph:
            base_word_idx = len(word_norm_ranges)
            for _, start_rel, end_rel in token_matches:
                global_start = seg_norm_start + start_rel
                global_end = seg_norm_start + end_rel
                word_norm_ranges.append((global_start, global_end))
            t_idx = 0
            for cnt in seg_word2ph:
                if t_idx < len(token_matches):
                    ph_to_word += [base_word_idx + t_idx] * cnt
                    t_idx += 1
                else:
                    ph_to_word += [-1] * cnt
            ph_to_char += [-1] * len(seg_phones)
        else:
            ph_to_char += [-1] * len(seg_phones)
            ph_to_word += [-1] * len(seg_phones)

        norm_offset += seg_norm_len
        orig_offset += len(seg_text)

    norm_text_agg = "".join(norm_text_parts)
    original_text_agg = "".join(text_segments)

    if len(segment_for_norm_index) < len(norm_text_agg):
        segment_for_norm_index.extend([-1] * (len(norm_text_agg) - len(segment_for_norm_index)))

    segment_for_word_index: List[int] = []
    for start_idx, end_idx in word_norm_ranges:
        seg_id = -1
        for idx in range(start_idx, min(end_idx, len(segment_for_norm_index))):
            cid = segment_for_norm_index[idx]
            if cid != -1:
                seg_id = cid
                break
        segment_for_word_index.append(seg_id)

    ph_to_segment: List[int] = []
    total_segments = len(segments_global)
    for char_idx, word_idx in zip(ph_to_char, ph_to_word):
        seg_id = -1
        if 0 <= char_idx < len(segment_for_norm_index):
            seg_id = segment_for_norm_index[char_idx]
        if seg_id == -1 and 0 <= word_idx < len(segment_for_word_index):
            seg_id = segment_for_word_index[word_idx]
        if seg_id is None or seg_id >= total_segments:
            seg_id = -1
        ph_to_segment.append(seg_id)

    return norm_text_agg, ph_to_segment, segments_global, original_text_agg, segments_raw_global
