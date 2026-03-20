import os
import re
import sys
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

now_dir = os.getcwd()
sys.path.append(now_dir)

from text.LangSegmenter import LangSegmenter
from text import cleaned_text_to_sequence
from text import chinese2
from text.cleaner import clean_text, clean_text_batch


PreparedTextSegmentPayload = Dict[str, object]
PreparedTextSegmentBatchItem = Tuple[str, str, str, bool]
_SegmentJob = Tuple[int, str, str, str]
_MULTISPACE_PATTERN = re.compile(r" {2,}")
_AUTO_ZH_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u4e00-\u9fff0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_AUTO_EN_FASTPATH_PATTERN = re.compile(
    r"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$"
)
_AUTO_ZH_FASTPATH_LATIN_PATTERN = re.compile(r"[A-Za-z\uff21-\uff3a\uff41-\uff5a]")
_AUTO_ZH_FASTPATH_JAKO_PATTERN = re.compile(r"[\u3040-\u30ff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_AUTO_JA_FASTPATH_ALLOWED_PATTERN = re.compile(
    r"^[\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff66-\uff9d0-9\s、，。！？,.!?…：；\-—~～/·]+$"
)
_AUTO_JA_DISTINCTIVE_PATTERN = re.compile(r"[\u3040-\u30ff\uff66-\uff9d]")
_AUTO_KO_DISTINCTIVE_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_YUE_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u4e00-\u9fff0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_JA_FASTPATH_ALLOWED_PATTERN = re.compile(
    r"^[\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d0-9\s、，。！？,.!?…：；\-—~～/·]+$"
)
_KO_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_DIRECT_FASTPATH_LATIN_PATTERN = re.compile(r"[A-Za-z\uff21-\uff3a\uff41-\uff5a]")
_WHITESPACE_TOKEN_PATTERN = re.compile(r"\S+\s*")
_TOKEN_STRIP_PUNCT_PATTERN = re.compile(r"^[、，。！？,.!?…：；\-—~～/·]+|[、，。！？,.!?…：；\-—~～/·]+$")
_TOKEN_HAS_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_PAYLOAD_CACHE_LOCK = threading.Lock()
_PAYLOAD_CACHE: "OrderedDict[PreparedTextSegmentBatchItem, List[PreparedTextSegmentPayload]]" = OrderedDict()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


_PAYLOAD_CACHE_MAX_ITEMS = max(0, int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_CACHE_MAX_ITEMS", "4096")))
_PAYLOAD_CACHE_ENABLED = _PAYLOAD_CACHE_MAX_ITEMS > 0 and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_CACHE", True)


def _normalize_spaces(text: str) -> str:
    return _MULTISPACE_PATTERN.sub(" ", str(text))


def _is_direct_zh_fast_path(language: str) -> bool:
    return _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_ZH_FASTPATH", True) and str(language) in {"zh", "all_zh"}


def _get_direct_language_fast_path(language: str) -> str | None:
    normalized = str(language)
    if normalized in {"all_yue"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_YUE_FASTPATH", True):
        return "yue"
    if normalized in {"all_ja"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_JA_FASTPATH", True):
        return "ja"
    if normalized in {"all_ko"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_KO_FASTPATH", True):
        return "ko"
    return None


def _can_use_direct_language_fast_path(text: str, language: str) -> str | None:
    target_language = _get_direct_language_fast_path(language)
    if target_language is None or not text:
        return None
    if target_language == "yue":
        if _YUE_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    if target_language == "ja":
        if _JA_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    if target_language == "ko":
        if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    return None


def _is_auto_zh_fast_path(text: str, language: str) -> bool:
    if str(language) not in {"auto", "auto_yue"}:
        return False
    if not _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_ZH_FASTPATH", False):
        return False
    if not text or not _AUTO_ZH_FASTPATH_ALLOWED_PATTERN.fullmatch(text):
        return False
    if _AUTO_ZH_FASTPATH_LATIN_PATTERN.search(text) or _AUTO_ZH_FASTPATH_JAKO_PATTERN.search(text):
        return False
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count > 0


def _get_auto_single_language_fast_path(text: str, language: str) -> str | None:
    normalized = str(language)
    if normalized not in {"auto", "auto_yue"}:
        return None
    if not _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_FASTPATH", True):
        return None
    if not text:
        return None
    if _AUTO_EN_FASTPATH_PATTERN.fullmatch(text):
        return "en"
    if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and _AUTO_KO_DISTINCTIVE_PATTERN.search(text):
        return "ko"
    if _AUTO_JA_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and _AUTO_JA_DISTINCTIVE_PATTERN.search(text):
        return "ja"
    if _AUTO_ZH_FASTPATH_ALLOWED_PATTERN.fullmatch(text):
        if _AUTO_ZH_FASTPATH_LATIN_PATTERN.search(text) or _AUTO_ZH_FASTPATH_JAKO_PATTERN.search(text):
            return None
        cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        if cjk_count > 0:
            return "yue" if normalized == "auto_yue" else "zh"
    return None


def _should_use_zh_fast_path(text: str, language: str) -> bool:
    return _is_direct_zh_fast_path(language) or _is_auto_zh_fast_path(text, language)


def _classify_whitespace_token(text: str, language: str) -> str | None:
    stripped = _TOKEN_STRIP_PUNCT_PATTERN.sub("", str(text).strip())
    if not stripped:
        return None
    if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(stripped) and _AUTO_KO_DISTINCTIVE_PATTERN.search(stripped):
        return "ko"
    if _AUTO_JA_FASTPATH_ALLOWED_PATTERN.fullmatch(stripped) and _AUTO_JA_DISTINCTIVE_PATTERN.search(stripped):
        return "ja"
    if _AUTO_EN_FASTPATH_PATTERN.fullmatch(stripped):
        return "en"
    return None


def _try_whitespace_mixed_fast_path(text: str, language: str) -> Tuple[List[str], List[str]] | None:
    if str(language) not in {"auto", "auto_yue"}:
        return None
    token_matches = list(_WHITESPACE_TOKEN_PATTERN.finditer(str(text)))
    if len(token_matches) <= 1:
        return None
    textlist: List[str] = []
    langlist: List[str] = []
    for match in token_matches:
        chunk = match.group(0)
        detected_lang = _classify_whitespace_token(chunk, language)
        if detected_lang is None:
            return None
        if langlist and langlist[-1] == detected_lang:
            textlist[-1] += chunk
            continue
        textlist.append(chunk)
        langlist.append(detected_lang)
    return textlist, langlist


def _build_zh_fast_path_payload(norm_text: str) -> List[PreparedTextSegmentPayload]:
    return [
        {
            "language": "zh",
            "phones": [],
            "word2ph": None,
            "norm_text": str(norm_text),
            "needs_g2pw": True,
        }
    ]


def _build_direct_language_payload(
    text: str,
    language: str,
    version: str,
) -> List[PreparedTextSegmentPayload]:
    phones, word2ph, norm_text = clean_text_segment(text, language, version)
    return [
        {
            "language": str(language).replace("all_", ""),
            "phones": phones,
            "word2ph": word2ph,
            "norm_text": norm_text,
            "needs_g2pw": False,
        }
    ]


def _estimate_payload_phones_len(payloads: Sequence[PreparedTextSegmentPayload]) -> int:
    total_phones_len = 0
    for payload in payloads:
        if bool(payload.get("needs_g2pw", False)):
            total_phones_len += max(0, len(str(payload.get("norm_text", ""))) * 2)
            continue
        total_phones_len += len(payload.get("phones", []))
    return int(total_phones_len)


def _build_segment_payload(
    *,
    language: str,
    phones: Sequence[int] | None,
    word2ph: Sequence[int] | None,
    norm_text: str,
    needs_g2pw: bool,
) -> PreparedTextSegmentPayload:
    return {
        "language": str(language),
        "phones": [] if phones is None else list(phones),
        "word2ph": None if word2ph is None else list(word2ph),
        "norm_text": str(norm_text),
        "needs_g2pw": bool(needs_g2pw),
    }


def _clone_payloads(payloads: Sequence[PreparedTextSegmentPayload]) -> List[PreparedTextSegmentPayload]:
    return [
        {
            "language": str(payload["language"]),
            "phones": list(payload["phones"]),
            "word2ph": None if payload["word2ph"] is None else list(payload["word2ph"]),
            "norm_text": str(payload["norm_text"]),
            "needs_g2pw": bool(payload.get("needs_g2pw", False)),
        }
        for payload in payloads
    ]


def _cache_get_payloads(item: PreparedTextSegmentBatchItem) -> List[PreparedTextSegmentPayload] | None:
    if not _PAYLOAD_CACHE_ENABLED:
        return None
    with _PAYLOAD_CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(item)
        if cached is None:
            return None
        _PAYLOAD_CACHE.move_to_end(item)
        return _clone_payloads(cached)


def _cache_store_payloads(
    item: PreparedTextSegmentBatchItem,
    payloads: Sequence[PreparedTextSegmentPayload],
) -> None:
    if not _PAYLOAD_CACHE_ENABLED:
        return
    cached_payloads = _clone_payloads(payloads)
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[item] = cached_payloads
        _PAYLOAD_CACHE.move_to_end(item)
        while len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX_ITEMS:
            _PAYLOAD_CACHE.popitem(last=False)


def _build_nonzh_segment_payloads_batch(
    jobs: Sequence[_SegmentJob],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    if not jobs:
        return payloads_by_index
    texts = [segment_text for _segment_index, segment_text, _segment_lang, _version in jobs]
    segment_lang = str(jobs[0][2])
    version = str(jobs[0][3])
    rows = clean_text_batch(texts, segment_lang, version)
    for (segment_index, _segment_text, segment_lang, version), (phones, word2ph, norm_text) in zip(jobs, rows):
        payloads_by_index[segment_index] = _build_segment_payload(
            language=segment_lang.replace("all_", ""),
            phones=cleaned_text_to_sequence(phones, version),
            word2ph=word2ph,
            norm_text=norm_text,
            needs_g2pw=False,
        )
    return payloads_by_index


def _build_zh_segment_payloads_batch(
    jobs: Sequence[_SegmentJob],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    if not jobs:
        return payloads_by_index
    norm_texts = chinese2.text_normalize_batch([segment_text for _, segment_text, _, _ in jobs])
    for (segment_index, _segment_text, _segment_lang, _version), norm_text in zip(jobs, norm_texts):
        payloads_by_index[segment_index] = _build_segment_payload(
            language="zh",
            phones=[],
            word2ph=None,
            norm_text=str(norm_text),
            needs_g2pw=True,
        )
    return payloads_by_index


def _build_segment_payloads_batch(
    jobs_by_language: Dict[Tuple[str, str], List[_SegmentJob]],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    for (normalized_language, _version), jobs in jobs_by_language.items():
        if normalized_language == "zh":
            payloads_by_index.update(_build_zh_segment_payloads_batch(jobs))
            continue
        payloads_by_index.update(_build_nonzh_segment_payloads_batch(jobs))
    return payloads_by_index


def split_text_by_language(text: str, language: str) -> Tuple[List[str], List[str]]:
    if _should_use_zh_fast_path(text, language):
        return [text], ["zh"]
    auto_language = _get_auto_single_language_fast_path(text, language)
    if auto_language is not None:
        return [text], [auto_language]
    direct_language = _can_use_direct_language_fast_path(text, language)
    if direct_language is not None:
        return [text], [direct_language]
    whitespace_fast_path = _try_whitespace_mixed_fast_path(text, language)
    if whitespace_fast_path is not None:
        return whitespace_fast_path
    textlist: List[str] = []
    langlist: List[str] = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                same_group = (tmp["lang"] == "en" and langlist[-1] == "en") or (
                    tmp["lang"] != "en" and langlist[-1] != "en"
                )
                if same_group:
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(language)
            textlist.append(tmp["text"])
    return textlist, langlist


def split_texts_by_language_batch(
    texts: Sequence[str],
    language: str,
) -> List[Tuple[List[str], List[str]]]:
    normalized_language = str(language)
    if not texts:
        return []
    if normalized_language in {"auto", "auto_yue"}:
        results: List[Tuple[List[str], List[str]] | None] = [None] * len(texts)
        fallback_pairs: List[Tuple[int, str]] = []
        for index, text in enumerate(texts):
            fast_path = _try_whitespace_mixed_fast_path(text, normalized_language)
            if fast_path is not None:
                results[index] = fast_path
            else:
                fallback_pairs.append((index, text))
        if not fallback_pairs:
            return [result for result in results if result is not None]
        fallback_rows = (
            LangSegmenter.getTextsBatch([text for _, text in fallback_pairs])
            if normalized_language == "auto"
            else LangSegmenter.getTextsBatch([text for _, text in fallback_pairs])
        )
        for (index, _text), items in zip(fallback_pairs, fallback_rows):
            if normalized_language == "auto_yue":
                textlist = []
                langlist = []
                for item in items:
                    item_lang = "yue" if item["lang"] == "zh" else item["lang"]
                    textlist.append(item["text"])
                    langlist.append(item_lang)
                results[index] = (textlist, langlist)
            else:
                results[index] = ([item["text"] for item in items], [item["lang"] for item in items])
        return [result for result in results if result is not None]
    if normalized_language == "all_zh":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "zh")
        ]
    if normalized_language == "all_yue":
        results: List[Tuple[List[str], List[str]]] = []
        for items in LangSegmenter.getTextsBatch(texts, "zh"):
            textlist: List[str] = []
            langlist: List[str] = []
            for item in items:
                item_lang = "yue" if item["lang"] == "zh" else item["lang"]
                textlist.append(item["text"])
                langlist.append(item_lang)
            results.append((textlist, langlist))
        return results
    if normalized_language == "all_ja":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "ja")
        ]
    if normalized_language == "all_ko":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "ko")
        ]
    if normalized_language == "en":
        return [([text], ["en"]) for text in texts]
    return [split_text_by_language(text, normalized_language) for text in texts]


def clean_text_segment(text: str, language: str, version: str) -> Tuple[List[int], Optional[List[int]], str]:
    normalized_language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, normalized_language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return list(phones), None if word2ph is None else list(word2ph), str(norm_text)


def _preprocess_text_segments_payload_impl(
    text: str,
    language: str,
    version: str,
    final: bool = False,
) -> List[PreparedTextSegmentPayload]:
    text = _normalize_spaces(text)
    if _should_use_zh_fast_path(text, language):
        norm_text = chinese2.text_normalize(text)
        if not final and max(0, len(norm_text) * 2) < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return _build_zh_fast_path_payload(norm_text)
    auto_language = _get_auto_single_language_fast_path(text, language)
    if auto_language is not None:
        if auto_language == "zh":
            norm_text = chinese2.text_normalize(text)
            if not final and max(0, len(norm_text) * 2) < 6:
                return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
            return _build_zh_fast_path_payload(norm_text)
        payloads = _build_direct_language_payload(text, auto_language, version)
        estimated_phones_len = len(payloads[0]["phones"])
        if not final and estimated_phones_len < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return payloads
    direct_language = _can_use_direct_language_fast_path(text, language)
    if direct_language is not None:
        payloads = _build_direct_language_payload(text, direct_language, version)
        estimated_phones_len = len(payloads[0]["phones"])
        if not final and estimated_phones_len < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return payloads
    textlist, langlist = split_text_by_language(text, language)
    payloads: List[PreparedTextSegmentPayload] = []
    total_phones_len = 0
    for segment_text, segment_lang in zip(textlist, langlist):
        normalized_language = segment_lang.replace("all_", "")
        if normalized_language == "zh":
            norm_text = chinese2.text_normalize(segment_text)
            phones = []
            word2ph = None
            needs_g2pw = True
            estimated_phones_len = max(0, len(norm_text) * 2)
        else:
            phones, word2ph, norm_text = clean_text_segment(segment_text, segment_lang, version)
            needs_g2pw = False
            estimated_phones_len = len(phones)
        payloads.append(
            {
                "language": normalized_language,
                "phones": phones,
                "word2ph": word2ph,
                "norm_text": norm_text,
                "needs_g2pw": needs_g2pw,
            }
        )
        total_phones_len += int(estimated_phones_len)

    if not final and total_phones_len < 6:
        return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)

    return payloads


def preprocess_text_segments_payload(
    text: str,
    language: str,
    version: str,
    final: bool = False,
) -> List[PreparedTextSegmentPayload]:
    item = (_normalize_spaces(str(text)), str(language), str(version), bool(final))
    cached = _cache_get_payloads(item)
    if cached is not None:
        return cached
    payloads = _preprocess_text_segments_payload_impl(*item)
    _cache_store_payloads(item, payloads)
    return _clone_payloads(payloads)


def preprocess_text_segments_payload_batch(
    items: Sequence[PreparedTextSegmentBatchItem],
) -> List[List[PreparedTextSegmentPayload]]:
    normalized_items = [
        (_normalize_spaces(str(text)), str(language), str(version), bool(final))
        for text, language, version, final in items
    ]
    results: List[List[PreparedTextSegmentPayload] | None] = [None] * len(normalized_items)
    duplicate_indices_by_root: Dict[int, List[int]] = {}
    unique_items: List[PreparedTextSegmentBatchItem] = []
    unique_result_indices: List[int] = []
    first_index_by_item: Dict[PreparedTextSegmentBatchItem, int] = {}

    for index, item in enumerate(normalized_items):
        cached = _cache_get_payloads(item)
        if cached is not None:
            results[index] = cached
            continue
        root_index = first_index_by_item.get(item)
        if root_index is not None:
            duplicate_indices_by_root.setdefault(root_index, []).append(index)
            continue
        first_index_by_item[item] = index
        unique_items.append(item)
        unique_result_indices.append(index)

    normalized_items = unique_items
    item_segment_indices: List[List[int]] = [[] for _ in normalized_items]
    jobs_by_language: Dict[Tuple[str, str], List[_SegmentJob]] = {}
    retry_items: List[PreparedTextSegmentBatchItem] = []
    retry_result_indices: List[int] = []
    next_segment_index = 0
    segment_specs_by_unique: List[List[Tuple[str, str]] | None] = [None] * len(normalized_items)
    split_batches_by_language: Dict[str, List[Tuple[int, str]]] = {}

    for unique_index, (text, language, version, final) in enumerate(normalized_items):
        segment_specs: List[Tuple[str, str]] = []
        if _should_use_zh_fast_path(text, language):
            segment_specs = [(text, "zh")]
        else:
            auto_language = _get_auto_single_language_fast_path(text, language)
            if auto_language is not None:
                segment_specs = [(text, auto_language)]
            else:
                direct_language = _can_use_direct_language_fast_path(text, language)
                if direct_language is not None:
                    segment_specs = [(text, direct_language)]
                else:
                    split_batches_by_language.setdefault(str(language), []).append((unique_index, text))

        if segment_specs:
            segment_specs_by_unique[unique_index] = segment_specs

    for language, pending_items in split_batches_by_language.items():
        split_results = split_texts_by_language_batch(
            [text for _unique_index, text in pending_items],
            language,
        )
        for (unique_index, _text), (textlist, langlist) in zip(pending_items, split_results):
            segment_specs_by_unique[unique_index] = list(zip(textlist, langlist))

    for unique_index, segment_specs in enumerate(segment_specs_by_unique):
        assert segment_specs is not None
        for segment_text, segment_lang in segment_specs:
            segment_index = next_segment_index
            next_segment_index += 1
            item_segment_indices[unique_index].append(segment_index)
            normalized_language = str(segment_lang).replace("all_", "")
            jobs_by_language.setdefault((normalized_language, version), []).append(
                (segment_index, str(segment_text), str(segment_lang), version)
            )

    payloads_by_segment = _build_segment_payloads_batch(jobs_by_language)

    for unique_index, segment_indices in enumerate(item_segment_indices):
        result_index = unique_result_indices[unique_index]
        payloads = [payloads_by_segment[segment_index] for segment_index in segment_indices]
        text, language, version, final = normalized_items[unique_index]
        if not final and _estimate_payload_phones_len(payloads) < 6:
            retry_items.append(("." + text, language, version, True))
            retry_result_indices.append(result_index)
            continue
        _cache_store_payloads((text, language, version, final), payloads)
        results[result_index] = payloads

    if retry_items:
        retry_results = preprocess_text_segments_payload_batch(retry_items)
        for result_index, payloads in zip(retry_result_indices, retry_results):
            results[result_index] = payloads

    for root_index, duplicate_indices in duplicate_indices_by_root.items():
        root_payloads = results[root_index]
        assert root_payloads is not None
        for duplicate_index in duplicate_indices:
            results[duplicate_index] = _clone_payloads(root_payloads)

    return [result if result is not None else [] for result in results]
