# reference: https://huggingface.co/spaces/Naozumi0512/Bert-VITS2-Cantonese-Yue/blob/main/text/chinese.py

import re
import cn2an
import ToJyutping

from text.symbols import punctuation
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

INITIALS = [
    "aa",
    "aai",
    "aak",
    "aap",
    "aat",
    "aau",
    "ai",
    "au",
    "ap",
    "at",
    "ak",
    "a",
    "p",
    "b",
    "e",
    "ts",
    "t",
    "dz",
    "d",
    "kw",
    "k",
    "gw",
    "g",
    "f",
    "h",
    "l",
    "m",
    "ng",
    "n",
    "s",
    "y",
    "w",
    "c",
    "z",
    "j",
    "ong",
    "on",
    "ou",
    "oi",
    "ok",
    "o",
    "uk",
    "ung",
]
INITIALS += ["sp", "spl", "spn", "sil"]


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    # text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def text_normalize(text):
    """
    Normalize Cantonese text and return character-level mapping.
    Returns: (normalized_text, char_map)
    Similar to Chinese, but for Cantonese.
    """
    original_text = text

    # Step 1: TextNormalizer (handles numbers, symbols, etc.)
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    text_after_norm = "".join(sentences)

    # Build mapping for TextNormalizer output
    map_step1 = []
    orig_idx = 0
    norm_idx = 0

    while norm_idx < len(text_after_norm) and orig_idx < len(text):
        if text_after_norm[norm_idx] == text[orig_idx]:
            # Direct match
            map_step1.append((orig_idx, min(orig_idx + 1, len(text))))
            norm_idx += 1
            orig_idx += 1
        elif orig_idx < len(text) and (text[orig_idx].isspace() or text[orig_idx] in punctuation):
            # Skip punctuation/space in original
            orig_idx += 1
        else:
            # Expansion occurred
            source_pos = orig_idx
            orig_idx += 1
            # Continue adding expanded chars until we find a match
            while norm_idx < len(text_after_norm) and (orig_idx >= len(text) or text_after_norm[norm_idx] != text[orig_idx]):
                map_step1.append((source_pos, min(source_pos + 1, len(text))))
                norm_idx += 1

    # Handle remaining normalized text
    while norm_idx < len(text_after_norm):
        if len(text) > 0:
            map_step1.append((len(text) - 1, len(text)))
        else:
            map_step1.append((0, 0))
        norm_idx += 1

    # Step 2: Replace punctuation
    text_final = ""
    map_final = []
    for i, ch in enumerate(text_after_norm):
        new_ch = replace_punctuation(ch)
        for _ in new_ch:
            text_final += _
            if i < len(map_step1):
                map_final.append(map_step1[i])
            else:
                map_final.append((-1, -1))

    return text_final, map_final


punctuation_set = set(punctuation)


def jyuping_to_initials_finals_tones(jyuping_syllables):
    initials_finals = []
    tones = []
    word2ph = []

    for syllable in jyuping_syllables:
        if syllable in punctuation:
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for punctuation
        elif syllable == "_":
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for underscore
        else:
            try:
                tone = int(syllable[-1])
                syllable_without_tone = syllable[:-1]
            except ValueError:
                tone = 0
                syllable_without_tone = syllable

            for initial in INITIALS:
                if syllable_without_tone.startswith(initial):
                    if syllable_without_tone.startswith("nga"):
                        initials_finals.extend(
                            [
                                syllable_without_tone[:2],
                                syllable_without_tone[2:] or syllable_without_tone[-1],
                            ]
                        )
                        # tones.extend([tone, tone])
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    else:
                        final = syllable_without_tone[len(initial) :] or initial[-1]
                        initials_finals.extend([initial, final])
                        # tones.extend([tone, tone])
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    break
    assert len(initials_finals) == len(tones)

    ###魔改为辅音+带音调的元音
    phones = []
    for a, b in zip(initials_finals, tones):
        if b not in [-1, 0]:  ###防止粤语和普通话重合开头加Y，如果是标点，不加。
            todo = "%s%s" % (a, b)
        else:
            todo = a
        if todo not in punctuation_set:
            todo = "Y%s" % todo
        phones.append(todo)

    # return initials_finals, tones, word2ph
    return phones, word2ph


def get_jyutping(text):
    jyutping_array = []
    punct_pattern = re.compile(r"^[{}]+$".format(re.escape("".join(punctuation))))

    syllables = ToJyutping.get_jyutping_list(text)

    for word, syllable in syllables:
        if punct_pattern.match(word):
            puncts = re.split(r"([{}])".format(re.escape("".join(punctuation))), word)
            for punct in puncts:
                if len(punct) > 0:
                    jyutping_array.append(punct)
        else:
            # match multple jyutping eg: liu4 ge3, or single jyutping eg: liu4
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable):
                raise ValueError(f"Failed to convert {word} to jyutping: {syllable}")
            jyutping_array.append(syllable)

    return jyutping_array


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


def g2p(text):
    # word2ph = []
    jyuping = get_jyutping(text)
    # print(jyuping)
    # phones, tones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    phones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    # phones = ["_"] + phones + ["_"]
    # tones = [0] + tones + [0]
    # word2ph = [1] + word2ph + [1]
    return phones, word2ph


if __name__ == "__main__":
    # text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = "佢個鋤頭太短啦。"
    text = text_normalize(text)
    # phones, tones, word2ph = g2p(text)
    phones, word2ph = g2p(text)
    # print(phones, tones, word2ph)
    print(phones, word2ph)
