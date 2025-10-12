from text import cleaned_text_to_sequence
import os
from typing import Dict, List, Sequence, Tuple
# if os.environ.get("version","v1")=="v1":
#     from text import chinese
#     from text.symbols import symbols
# else:
#     from text import chinese2 as chinese
#     from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

Span = Tuple[int, int]


def _coerce_char_spans(char_map: Sequence, original_len: int, norm_len: int) -> List[Span]:
    spans: List[Span] = []
    for entry in char_map or []:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            start, end = entry
            start = int(start) if start is not None else -1
            end = int(end) if end is not None else start
            if start < 0:
                spans.append((-1, -1))
                continue
            end = max(start, end)
            spans.append((start, end))
        elif isinstance(entry, int):
            if original_len == 0:
                spans.append((0, 0))
            else:
                idx = max(0, min(entry, original_len - 1))
                spans.append((idx, idx + 1))
        else:
            spans.append((-1, -1))

    if len(spans) < norm_len:
        spans.extend([(-1, -1)] * (norm_len - len(spans)))
    elif len(spans) > norm_len:
        spans = spans[:norm_len]

    return spans


def _build_segment_map(norm_text: str, char_spans: Sequence[Span], original_len: int) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    if not norm_text or not char_spans:
        return segments

    idx = 0
    while idx < len(norm_text):
        ch = norm_text[idx]
        span = char_spans[idx]
        if ch.isspace() or span == (-1, -1):
            idx += 1
            continue

        seg_start = idx
        current_span = span
        idx += 1
        while idx < len(norm_text):
            if norm_text[idx].isspace() or char_spans[idx] != current_span:
                break
            idx += 1

        orig_start = current_span[0]
        orig_end = max(current_span[1], orig_start)

        # Extend end to the start of the next different span if available
        lookahead_idx = idx
        while lookahead_idx < len(char_spans):
            next_span = char_spans[lookahead_idx]
            if next_span == current_span or next_span == (-1, -1):
                lookahead_idx += 1
                continue
            orig_end = max(orig_start, next_span[0])
            break
        else:
            orig_end = max(orig_start, original_len)

        segments.append({
            "norm_start": seg_start,
            "norm_end": idx,
            "orig_start": orig_start,
            "orig_end": orig_end,
            "text": norm_text[seg_start:idx],
        })

    return segments


def _merge_segments_by_span(segments: Sequence[Dict[str, object]], norm_text: str) -> List[Dict[str, object]]:
    if not segments:
        return []

    merged: List[Dict[str, object]] = []
    current = dict(segments[0])

    for seg in segments[1:]:
        same_span = (
            seg["orig_start"] == current["orig_start"]
            and seg["orig_end"] == current["orig_end"]
            and current["orig_start"] != -1
        )
        between = norm_text[current["norm_end"]:seg["norm_start"]]
        if same_span and all(ch.isspace() for ch in between):
            current["text"] += between + seg["text"]
            current["norm_end"] = seg["norm_end"]
            continue

        merged.append(current)
        current = dict(seg)

    merged.append(current)
    return merged

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language, version=None):
    """
    Clean and normalize text, returning phones, word2ph, normalized text, and mapping details.
    Returns: (phones, word2ph, norm_text, mapping)
    mapping:
      - 'char_spans': per-character spans (orig_start, orig_end) in the original text
      - 'segments': list of contiguous non-space normalized segments with original spans
    """
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)

    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])

    # Get normalized text and character mapping
    if hasattr(language_module, "text_normalize"):
        normalize_result = language_module.text_normalize(text)
        if isinstance(normalize_result, tuple) and len(normalize_result) == 2:
            # New format: (norm_text, char_map)
            norm_text, char_map = normalize_result
        else:
            # Old format (should not happen after our changes, but keep for safety)
            norm_text = normalize_result
            char_map = [(idx, idx + 1) for idx in range(len(norm_text))]  # 1:1 mapping as fallback
    else:
        norm_text = text
        char_map = [(idx, idx + 1) for idx in range(len(text))]  # 1:1 mapping

    if language == "zh" or language == "yue":  ##########
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        # Try per-language word2ph helpers
        if hasattr(language_module, "g2p_with_word2ph"):
            try:
                phones, word2ph = language_module.g2p_with_word2ph(norm_text, keep_punc=False)
            except Exception:
                phones = language_module.g2p(norm_text)
                word2ph = None
        else:
            phones = language_module.g2p(norm_text)
            word2ph = None
        if language == "en" and len(phones) < 4:
            phones = [","] + phones
    phones = ["UNK" if ph not in symbols else ph for ph in phones]

    char_spans = _coerce_char_spans(char_map, len(text), len(norm_text))
    segments_raw = _build_segment_map(norm_text, char_spans, len(text))
    segments = _merge_segments_by_span(segments_raw, norm_text)

    mapping = {
        "char_spans": char_spans,
        "segments": segments,
        "segments_raw": segments_raw,
    }

    return phones, word2ph, norm_text, mapping


def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    norm_result = language_module.text_normalize(text)
    if isinstance(norm_result, tuple) and len(norm_result) == 2:
        norm_text, char_map = norm_result
    else:
        norm_text = norm_result
        char_map = [(idx, idx + 1) for idx in range(len(norm_text))]

    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    char_spans = _coerce_char_spans(char_map, len(text), len(norm_text))
    segments_raw = _build_segment_map(norm_text, char_spans, len(text))
    mapping = {
        "char_spans": char_spans,
        "segments": _merge_segments_by_span(segments_raw, norm_text),
        "segments_raw": segments_raw,
    }
    return new_ph, phones[1], norm_text, mapping


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones, _, _, _ = clean_text(text, language, version)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
