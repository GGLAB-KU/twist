#!/usr/bin/env python3
"""
Clean the CX-corpora EN→TR dataset – no temp files.

Defaults
--------
input  : inputs/cx-corpora.en2tr.text.json
terms  : inputs/terms.xlsx
cats   : inputs/wikipedia_id_category.json
output : outputs/final.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import stanza
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════
# Regex patterns
# ════════════════════════════════════════════════════════════════════════
RE_DOUBLE_COMMA = re.compile(r',\s*,')
RE_UP_ARROW = re.compile('↑')
RE_AMP = re.compile('&')
RE_DISPLAYSTYLE = re.compile('displaystyle')
RE_BRACKET_REF = re.compile(r'\[\d+\]')


# ════════════════════════════════════════════════════════════════════════
# Logging helper
# ════════════════════════════════════════════════════════════════════════
def _log_rows(log: logging.Logger, step: int, desc: str,
              before: int, after: int) -> int:
    """Log rows removed & remaining; return next step number."""
    removed = before - after
    log.info("Step %-2d: %-50s | removed: %-6d remaining: %-6d",
             step, desc, removed, after)
    return step + 1


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def load_and_fix_json(path: Path) -> pd.DataFrame:
    """Load raw JSON (with possible stray “,,”) and flatten to DataFrame."""
    text = RE_DOUBLE_COMMA.sub(',', path.read_text(encoding="utf-8"))
    return pd.json_normalize(json.loads(text))


# ════════════════════════════════════════════════════════════════════════
# Stage 1 · basic row / column cleaning
# ════════════════════════════════════════════════════════════════════════
def clean_stage_one(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    step = 0
    step = _log_rows(log, step, "initial rows", 0, len(df))

    df = df.drop(columns=['sourceLanguage', 'targetLanguage', 'mt'])

    before = len(df)
    df = df.dropna(subset=['source.content', 'target.content'])
    step = _log_rows(log, step, "drop empty source/target", before, len(df))

    for col in ['source.content', 'target.content']:
        before = len(df)
        df = df.drop_duplicates(subset=[col], keep='last')
        step = _log_rows(log, step, f"deduplicate on {col}", before, len(df))

    # identical texts
    df['mt.content'] = df['mt.content'].fillna('')
    before = len(df)
    df = df[(df['source.content'] != df['target.content']) &
            (df['target.content'] != df['mt.content'])]
    step = _log_rows(log, step, "remove exact matches", before, len(df))

    # length & dot filters
    def _ok(s: pd.Series) -> pd.Series:
        return (s.str.len() > 200) & (s.str.count(r'\.') > 2)

    before = len(df)
    df = df[_ok(df['source.content']) & _ok(df['target.content'])]
    step = _log_rows(log, step, "len>200 & ≥3 dots", before, len(df))

    # length ratio ±20 %
    before = len(df)
    ratio = df['source.content'].str.len() / df['target.content'].str.len()
    df = df[ratio.between(1 / 1.2, 1.2)]
    step = _log_rows(log, step, "length-ratio ±20 %", before, len(df))

    # illegal substrings
    for pat, label in [(RE_UP_ARROW, "↑ arrow"),
                       (RE_AMP, "& ampersand"),
                       (RE_DISPLAYSTYLE, "displaystyle")]:
        before = len(df)
        for c in ['source.content', 'target.content', 'mt.content']:
            df = df[~df[c].str.contains(pat)]
        step = _log_rows(log, step, f"remove '{label}'", before, len(df))

    return df


# ════════════════════════════════════════════════════════════════════════
# Stage 2 · terminology enrichment
# ════════════════════════════════════════════════════════════════════════
def compile_term_regex(path: Path) -> re.Pattern:
    terms = pd.read_excel(path)['term'].dropna().str.lower().tolist()
    terms += [t[:-1] + 'ies' if t.endswith('y') else t for t in terms]
    esc = map(re.escape, set(terms))
    return re.compile(r'\b(?:' + '|'.join(sorted(esc)) + r')(?:s|es)?\b')


def enrich_with_terms(df: pd.DataFrame, term_re: re.Pattern,
                      log: logging.Logger, step: int) -> Tuple[pd.DataFrame, int]:
    tqdm.pandas(desc="Scanning terms")
    df['terms'] = df['source.content'].str.lower().progress_apply(term_re.findall)
    df['num_terms'] = df['terms'].apply(len)
    df['num_unique_terms'] = df['terms'].apply(lambda x: len(set(x)))

    before = len(df)
    df = df[df['num_unique_terms'] > 3]
    step = _log_rows(log, step, "num_unique_terms > 3", before, len(df))
    return df, step


# ════════════════════════════════════════════════════════════════════════
# Stage 3 · tokenisation & sentence alignment
# ════════════════════════════════════════════════════════════════════════
_STANZA: dict[str, stanza.Pipeline] = {}


def nlp(lang: str) -> stanza.Pipeline:
    if lang not in _STANZA:
        _STANZA[lang] = stanza.Pipeline(
            lang=lang, processors='tokenize', logging_level='ERROR',
            download_method=None)
    return _STANZA[lang]


def _tok(series: pd.Series, lang: str, desc: str) -> Tuple[pd.Series, pd.Series]:
    sents, counts = [], []
    for txt in tqdm(series.fillna(''), desc=desc):
        doc = nlp(lang)(txt)
        L = [s.text for s in doc.sentences]
        sents.append(L)
        counts.append(len(L))
    return pd.Series(sents, index=series.index), pd.Series(counts, index=series.index)


def clean_stage_three(df: pd.DataFrame, log: logging.Logger, step: int) -> Tuple[pd.DataFrame, int]:
    for c in ['source.content', 'target.content', 'mt.content']:
        df[c] = df[c].fillna('').str.replace(RE_BRACKET_REF, '', regex=True)

    df['src_sent'], df['num_source_sentences'] = _tok(df['source.content'], 'en', 'Tokenise EN')
    df['tgt_sent'], df['num_target_sentences'] = _tok(df['target.content'], 'tr', 'Tokenise TR')
    df['mt_sent'], df['num_mt_sentences'] = _tok(df['mt.content'], 'tr', 'Tokenise MT')

    join = lambda L: '<SENTENCE>'.join(s.capitalize() for s in L)
    df['source_text'] = df['src_sent'].apply(join)
    df['target_text'] = df['tgt_sent'].apply(join)
    df['mt_text'] = df['mt_sent'].apply(join)

    # align sentence counts
    before = len(df)
    df = df[df['num_source_sentences'] == df['num_target_sentences']]
    step = _log_rows(log, step, "align sentence counts", before, len(df))

    # ensure MT differs
    before = len(df)
    df = df[(df['source_text'] != df['mt_text']) &
            (df['target_text'] != df['mt_text'])]
    step = _log_rows(log, step, "MT ≠ SRC/TGT", before, len(df))

    df['num_sentences'] = df['num_source_sentences']
    return df, step


# ════════════════════════════════════════════════════════════════════════
# Stage 4 · attach categories
# ════════════════════════════════════════════════════════════════════════
def attach_categories(df: pd.DataFrame, cats_path: Path,
                      log: logging.Logger, step: int) -> Tuple[pd.DataFrame, int]:
    cat_map: dict[str, str] = json.loads(cats_path.read_text(encoding="utf-8"))
    before = len(df)
    df = df[df['id'].isin(cat_map)]
    step = _log_rows(log, step, "retain rows with category", before, len(df))
    df['category'] = df['id'].map(cat_map)
    return df, step


# ════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════
FINAL_COLUMNS = [
    "wikipedia_id", "mt.engine",
    "terms", "num_terms", "num_unique_terms",
    "num_source_sentences", "num_target_sentences", "num_mt_sentences",
    "source_text", "mt_text", "target_text",
    "num_sentences", "category",
]


def run_pipeline(inp: Path, terms: Path, cats: Path, out: Path) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("cx-clean")

    df = load_and_fix_json(inp)
    step = 0

    # Stage 1
    df = clean_stage_one(df, log)

    # Stage 2
    df, step = enrich_with_terms(df, compile_term_regex(terms), log, step)

    # Stage 3
    df, step = clean_stage_three(df, log, step)

    # Stage 4
    df, step = attach_categories(df, cats, log, step)

    # rename id → wikipedia_id & reorder
    df.rename(columns={'id': 'wikipedia_id'}, inplace=True)
    df = df[FINAL_COLUMNS]

    _log_rows(log, step, "final dataset", 0, len(df))

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    log.info("Wrote wikipedia dataset → %s", out)
    return df


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CX-corpora (EN→TR)")
    p.add_argument("--input", default=Path("inputs/cx-corpora.en2tr.text.json"),
                   type=Path, help="CX-corpora JSON file")
    p.add_argument("--terms", default=Path("inputs/terms.xlsx"),
                   type=Path, help="Terminology list (Excel)")
    p.add_argument("--cats", default=Path("inputs/wikipedia_id_category.json"),
                   type=Path, help="Wikipedia-ID → category map (JSON)")
    p.add_argument("-o", "--output", default=Path("outputs/wikipedia_dataset_V1.xlsx"),
                   type=Path, help="Write cleaned data to this Excel file")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    run_pipeline(args.input, args.terms, args.cats, args.output)
