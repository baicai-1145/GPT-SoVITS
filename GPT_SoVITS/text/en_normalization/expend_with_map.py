# Modified version of expend.py that tracks character mapping
# This enables accurate mapping from normalized text back to original text

from __future__ import print_function
import re
import inflect
import unicodedata

# Import all the patterns and maps from original expend.py
from .expend import (
    measurement_map,
    _inflect,
    _ordinal_number_re,
    _comma_number_re,
    _time_re,
    _measurement_re,
    _pounds_re_start,
    _pounds_re_end,
    _dollars_re_start,
    _dollars_re_end,
    _decimal_number_re,
    _fraction_re,
    _ordinal_re,
    _number_re,
    _convert_ordinal,
    _remove_commas,
    _expand_time,
    _expand_measurement,
    _expand_pounds,
    _expand_dollars,
    _expand_decimal_number,
    _expend_fraction,
    _expand_ordinal,
    _expand_number,
)


def apply_regex_with_map(pattern, replacement_func, text, char_map):
    """
    Apply regex substitution while tracking character mapping.

    Args:
        pattern: Compiled regex pattern
        replacement_func: Function that takes match object and returns replacement string
        text: Current text
        char_map: Current mapping list where char_map[i] = original position for text[i]

    Returns:
        (new_text, new_char_map)
    """
    new_text = []
    new_map = []
    pos = 0

    for match in pattern.finditer(text):
        # Add text before match (unchanged)
        for i in range(pos, match.start()):
            new_text.append(text[i])
            new_map.append(char_map[i])

        # Add replacement text - all chars map to start of original match
        replacement = replacement_func(match)
        source_pos = char_map[match.start()] if match.start() < len(char_map) else (char_map[-1] if char_map else 0)

        for char in replacement:
            new_text.append(char)
            new_map.append(source_pos)

        pos = match.end()

    # Add remaining text
    for i in range(pos, len(text)):
        new_text.append(text[i])
        new_map.append(char_map[i])

    return ''.join(new_text), new_map


def apply_simple_sub_with_map(pattern, replacement, text, char_map):
    """
    Apply simple string substitution while tracking character mapping.

    Args:
        pattern: String or regex pattern to match
        replacement: Replacement string
        text: Current text
        char_map: Current mapping list

    Returns:
        (new_text, new_char_map)
    """
    if isinstance(pattern, str):
        pattern = re.compile(re.escape(pattern))

    def replace_func(m):
        return replacement

    return apply_regex_with_map(pattern, replace_func, text, char_map)


def normalize_with_map(text):
    """
    Normalize English text and return character-level mapping.

    Returns:
        (normalized_text, char_map)
        where char_map[i] = original position for normalized_text[i]
    """
    original_text = text

    # Initialize mapping: 1:1 at start
    char_map = list(range(len(text)))

    # Step 1: Ordinal number conversion
    text, char_map = apply_regex_with_map(_ordinal_number_re, _convert_ordinal, text, char_map)

    # Step 2: Minus sign
    def replace_minus(m):
        return " minus "
    text, char_map = apply_regex_with_map(re.compile(r"(?<!\d)-|-(?!\d)"), replace_minus, text, char_map)

    # Step 3: Comma removal in numbers
    text, char_map = apply_regex_with_map(_comma_number_re, _remove_commas, text, char_map)

    # Step 4: Time expansion
    text, char_map = apply_regex_with_map(_time_re, _expand_time, text, char_map)

    # Step 5: Measurement expansion
    text, char_map = apply_regex_with_map(_measurement_re, _expand_measurement, text, char_map)

    # Step 6-7: Pounds expansion
    text, char_map = apply_regex_with_map(_pounds_re_start, _expand_pounds, text, char_map)
    text, char_map = apply_regex_with_map(_pounds_re_end, _expand_pounds, text, char_map)

    # Step 8-9: Dollars expansion
    text, char_map = apply_regex_with_map(_dollars_re_start, _expand_dollars, text, char_map)
    text, char_map = apply_regex_with_map(_dollars_re_end, _expand_dollars, text, char_map)

    # Step 10: Decimal number expansion
    text, char_map = apply_regex_with_map(_decimal_number_re, _expand_decimal_number, text, char_map)

    # Step 11: Fraction expansion
    text, char_map = apply_regex_with_map(_fraction_re, _expend_fraction, text, char_map)

    # Step 12: Ordinal expansion
    text, char_map = apply_regex_with_map(_ordinal_re, _expand_ordinal, text, char_map)

    # Step 13: Number expansion
    text, char_map = apply_regex_with_map(_number_re, _expand_number, text, char_map)

    # Step 14: Unicode normalization (NFD + strip accents)
    # This is tricky - we need to handle character deletions
    new_text = []
    new_map = []
    for i, char in enumerate(text):
        normalized = unicodedata.normalize("NFD", char)
        for nc in normalized:
            if unicodedata.category(nc) != "Mn":  # Not a combining mark
                new_text.append(nc)
                new_map.append(char_map[i] if i < len(char_map) else 0)
    text = ''.join(new_text)
    char_map = new_map

    # Step 15: Percent sign
    text, char_map = apply_simple_sub_with_map("%", " percent", text, char_map)

    # Step 16: Character filtering - only keep allowed characters
    # Pattern: [^ A-Za-z'.,?!\-]
    allowed_chars = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'.,?!-")
    new_text = []
    new_map = []
    for i, char in enumerate(text):
        if char in allowed_chars:
            new_text.append(char)
            new_map.append(char_map[i] if i < len(char_map) else 0)
    text = ''.join(new_text)
    char_map = new_map

    # Step 17: Abbreviation expansion
    text, char_map = apply_regex_with_map(
        re.compile(r"(?i)i\.e\."),
        lambda m: "that is",
        text, char_map
    )
    text, char_map = apply_regex_with_map(
        re.compile(r"(?i)e\.g\."),
        lambda m: "for example",
        text, char_map
    )

    # Step 18: Split consecutive uppercase letters (e.g., "GPT" -> "G P T")
    # Pattern: (?<!^)(?<![\s])([A-Z])  => Replace with " \1"
    # Special handling: the inserted space should map to the previous character
    pattern_uppercase = re.compile(r"(?<!^)(?<![\s])([A-Z])")
    new_text = []
    new_map = []
    pos = 0

    for match in pattern_uppercase.finditer(text):
        # Add text before match (unchanged)
        for i in range(pos, match.start()):
            new_text.append(text[i])
            new_map.append(char_map[i])

        # Add space - map to previous character position
        prev_pos = char_map[match.start() - 1] if match.start() > 0 else char_map[0]
        new_text.append(" ")
        new_map.append(prev_pos)

        # Add the uppercase letter - map to itself
        new_text.append(match.group(1))
        new_map.append(char_map[match.start()])

        pos = match.end()

    # Add remaining text
    for i in range(pos, len(text)):
        new_text.append(text[i])
        new_map.append(char_map[i])

    text = ''.join(new_text)
    char_map = new_map

    return text, char_map
