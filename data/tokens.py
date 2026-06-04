"""
dataset/tokens.py — Tokenisation + curriculum assembly du mix 50B tokens
─────────────────────────────────────────────────────────────────────────
Pipeline :
  1. Pre-flight  : vérifie l'accès à chaque dataset (1 doc) avant tout download
  2. Download    : pipeline 3 threads → chunks binaires de ~1B tokens
                   ┌─ Thread 1 (producer)  : stream ModelScope → raw_q
                   ├─ Thread 2 (tokenizer) : raw_q → batch Rust → tok_q
                   └─ Thread 3 (main)      : tok_q → ChunkWriter
                   Gain typique : ×3–6 vs version mono-thread
                   (overlap réseau + CPU ; tokenizer Rust libère le GIL)
                   chunks/{dataset_name}/chunk_000.bin, chunk_001.bin, ...
  3. Assembly    : fichiers de 10B tokens (~20 GB en uint16) :
                   pretrain_data_000.bin … pretrain_data_NNN.bin
                   Curriculum en 2 phases :

       Phase 1 (0 → 30 % tokens)  : Cosmopedia-v2 seul, séquentiel
       Phase 2 (30 → 100 % tokens): Tous datasets shufflés ensemble

  Chaque fichier est uploadé sur HuggingFace Hub dès sa clôture.

  Source                                          Phase   Volume   Langue
  ──────────────────────────────────────────────────────────────────────
  HuggingFaceTB/smollm-corpus [cosmopedia-v2]       1     15.0B    EN
  nv-community/Nemotron-CC-v2.1 [Non-Synth HQ]      2     10.0B    EN
  tokyotech-llm/swallow-code-v2 [Python, no-JP]     2      7.0B    Code
  HuggingFaceFW/finephrase [all]                     2      7.0B    EN
  nv-community/Nemotron-CC-Math-v1 [4plus]           2      6.0B    EN
  nv-community/Nemotron-CC-v2.1 [High-Synth]         2      5.0B    EN
  ──────────────────────────────────────────────────────────────────────
  TOTAL                                                    50.0B

Tokenizer : HuggingFaceTB/cosmo2-tokenizer (vocab 49 152, uint16)
Reprise   : chunks/resume.json
Usage     : python tokens.py [--reset] [--skip-assembly] [--only-assembly]
            python tokens.py --no-preflight
            python tokens.py --batch-size 4096
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("✗ tqdm non installé : pip install tqdm")

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("✗ transformers non installé : pip install transformers")

# ── Tokens & identifiants ──────────────────────────────────────────────────────

TOKENIZER_ID    = "HuggingFaceTB/cosmo2-tokenizer"
TOKENIZER_LOCAL = Path("tokenizer")

DTYPE  = np.uint32   # écrasé dans load_tokenizer()
EOS_ID: int = 0      # écrasé dans load_tokenizer()

HF_TOKEN         = os.environ.get("HF_TOKEN", "")
MODELSCOPE_TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")

# ── Chemins ────────────────────────────────────────────────────────────────────

CHUNKS_DIR  = Path("chunks")
RESUME_FILE = CHUNKS_DIR / "resume.json"
OUT_PREFIX  = "pretrain_data"

# ── Paramètres de découpe ──────────────────────────────────────────────────────

CHUNK_TOKENS   = 1_000_000_000   # 1B tokens par chunk intermédiaire
WRITE_BUF_TOKS = 25_000_000      # buffer d'écriture interne
SPLIT_TOKENS   = 10_000_000_000  # taille d'un fichier final

# Seuil minimal pour considérer un dataset comme "complété" (95 % de la cible).
MIN_COMPLETION_RATIO = 0.95

# Longueur moyenne estimée par doc — utilisée pour reconstruire doc_offset
# quand on recalcule l'état depuis le disque.
_AVG_TOKS_PER_DOC: dict[str, int] = {
    "cosmopedia_v2"      : 430,
    "nemotron_non_synth" : 800,
    "swallow_code"       : 600,
    "finephrase"         : 350,
    "nemotron_math"      : 500,
    "nemotron_high_synth": 600,
}

HF_DATASET_REPO = "silyan/Naylis1-1B"

# ── Pipeline producer-consumer ─────────────────────────────────────────────────

TOKENIZER_BATCH_SIZE = 512   # surchargeable via --batch-size

RAW_Q_MAX = 8_000
TOK_Q_MAX =   512

_SENTINEL = object()

# ── Curriculum ─────────────────────────────────────────────────────────────────

P1_END_FRAC   = 0.30
P3_START_FRAC = 1.00

REPLAY_DATASETS    = set()
REPLAY_RATIO_START = 0.0
REPLAY_RATIO_END   = 0.0

TEXT_FIELDS = ["text", "code", "content", "document", "passage", "input", "problem"]

# Détection japonais (hiragana + katakana + CJK)
# \U00020000 (8 hex) pour CJK Extension B (plan supplémentaire)
import re as _re
_JP_RE = _re.compile(
    r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF'
    r'\uF900-\uFAFF\U00020000-\U0002A6DF]'
)

FINEWEB2_MIN_QUALITY = 0.7

# ── Définition des datasets ────────────────────────────────────────────────────

DATASETS: list[dict] = [
    {
        "name"        : "cosmopedia_v2",
        "hf_id"       : "HuggingFaceTB/smollm-corpus",
        "subset"      : "cosmopedia-v2",
        "split"       : "train",
        "token_target": 15_000_000_000,
        "phase"       : 1,
    },
    {
        "name"        : "nemotron_non_synth",
        "hf_id"       : "nv-community/Nemotron-CC-v2.1",
        "subset"      : "High-Quality",
        "split"       : "train",
        "token_target": 10_000_000_000,
        "phase"       : 2,
    },
    {
        "name"        : "swallow_code",
        "hf_id"       : "tokyotech-llm/swallow-code-v2",
        "subset"      : "swallowcode-v2",
        "split"       : "train",
        "token_target": 7_000_000_000,
        "phase"       : 2,
        "lang_filter" : "python",
        "skip_jp"     : True,
        "safe_cast"   : True,   # patch schéma JSONL incohérent (lint_report)
    },
    {
        "name"        : "finephrase",
        "hf_id"       : "HuggingFaceFW/finephrase",
        "subset"      : "all",
        "split"       : "train",
        "token_target": 7_000_000_000,
        "phase"       : 2,
    },
    {
        "name"        : "nemotron_math",
        "hf_id"       : "nv-community/Nemotron-CC-Math-v1",
        "subset"      : "4plus",
        "split"       : "train",
        "token_target": 6_000_000_000,
        "phase"       : 2,
    },
    {
        "name"        : "nemotron_high_synth",
        "hf_id"       : "nv-community/Nemotron-CC-v2.1",
        "subset"      : "High-Quality-Synthetic",
        "split"       : "train",
        "token_target": 5_000_000_000,
        "phase"       : 2,
    },
]

# ── Patch schéma JSONL incohérent ─────────────────────────────────────────────

def _apply_safe_cast_patch() -> None:
    """
    Corrige le bug de cast de schéma dans la bibliothèque `datasets` pour les
    datasets JSONL dont certains shards ont un type incompatible avec le schéma
    déclaré (ex : swallow-code-v2, champ lint_report : null vs struct).
    """
    from datasets.packaged_modules.json import json as _hf_json
    from datasets import table as _hf_table
    import datasets.iterable_dataset as _hf_iter

    if not getattr(_hf_json.Json._cast_table, '_safe_cast_patched', False):
        _orig = _hf_json.Json._cast_table
        def _safe(self, pa_table, json_field_paths=None):
            try:
                return _orig(self, pa_table, json_field_paths=json_field_paths)
            except TypeError:
                return pa_table
        _safe._safe_cast_patched = True
        _hf_json.Json._cast_table = _safe

    if not getattr(_hf_table.cast_table_to_features, '_safe_cast_patched', False):
        _orig2 = _hf_table.cast_table_to_features
        def _safe2(table, features):
            try:
                return _orig2(table, features)
            except TypeError:
                return table
        _safe2._safe_cast_patched = True
        _hf_table.cast_table_to_features = _safe2
        _hf_iter.cast_table_to_features  = _safe2


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def load_tokenizer() -> AutoTokenizer:
    global EOS_ID, DTYPE

    src = TOKENIZER_LOCAL if TOKENIZER_LOCAL.exists() else TOKENIZER_ID
    print(f"Chargement tokenizer {'(local)' if TOKENIZER_LOCAL.exists() else TOKENIZER_ID}…")

    try:
        tok = AutoTokenizer.from_pretrained(
            str(src), trust_remote_code=True, token=HF_TOKEN or None,
        )
    except Exception as e:
        sys.exit(f"✗ Impossible de charger le tokenizer : {e}")

    if not TOKENIZER_LOCAL.exists():
        TOKENIZER_LOCAL.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(TOKENIZER_LOCAL))
        print(f"  ✓ Tokenizer sauvegardé → {TOKENIZER_LOCAL}/")

    EOS_ID = tok.eos_token_id
    if EOS_ID is None:
        EOS_ID = tok.convert_tokens_to_ids("<|endoftext|>")

    vocab_size    = len(tok)
    DTYPE         = np.uint16 if vocab_size <= 65535 else np.uint32
    dtype_name    = "uint16" if DTYPE == np.uint16 else "uint32"
    bytes_per_tok = 2 if DTYPE == np.uint16 else 4
    total_gb      = 50e9 * bytes_per_tok / 1e9

    print(f"  vocab total  : {vocab_size:,}")
    print(f"  EOS          : {tok.eos_token!r} = {EOS_ID}")
    print(f"  dtype        : {dtype_name}  ({bytes_per_tok} octets/token, ~{total_gb:.0f} GB pour 50B tokens)")

    return tok


# ── Accès aux datasets ─────────────────────────────────────────────────────────

def _get_text(row) -> str:
    if isinstance(row, str):
        return row
    for f in TEXT_FIELDS:
        if f in row and row[f]:
            return str(row[f])
    vals = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 10]
    return vals[0] if vals else ""


def _load_hf_stream(cfg: dict, doc_offset: int = 0):
    from datasets import load_dataset as hf_load
    kwargs: dict = dict(path=cfg["hf_id"], split=cfg["split"], streaming=True)
    if cfg.get("subset"):
        kwargs["name"] = cfg["subset"]
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    ds = hf_load(**kwargs)
    if doc_offset:
        ds = ds.skip(doc_offset)
    return ds


def stream_dataset(cfg: dict, doc_offset: int = 0) -> Iterator[str]:
    if cfg.get("safe_cast"):
        _apply_safe_cast_patch()

    lang        = cfg.get("lang_filter", "").strip().lower()
    min_quality = cfg.get("quality_filter")
    skip_jp     = cfg.get("skip_jp", False)

    def _pass(row) -> bool:
        if lang and isinstance(row, dict):
            row_lang = (row.get("language") or row.get("lang")
                        or row.get("programming_language") or "")
            if str(row_lang).strip().lower() != lang:
                return False
        if min_quality is not None and isinstance(row, dict):
            score = row.get("quality_score")
            if score is not None and float(score) < min_quality:
                return False
        return True

    def _no_jp(text: str) -> bool:
        return not _JP_RE.search(text)

    if cfg.get("use_hf"):
        try:
            ds = _load_hf_stream(cfg, doc_offset)
            for row in ds:
                if _pass(row):
                    t = _get_text(row)
                    if t and (not skip_jp or _no_jp(t)):
                        yield t
        except ImportError:
            sys.exit("✗ pip install datasets")
    else:
        try:
            from modelscope import MsDataset
        except ImportError:
            sys.exit("✗ pip install modelscope")
        kwargs: dict = dict(dataset_name=cfg["hf_id"], split=cfg["split"])
        if cfg.get("subset"):
            kwargs["subset_name"] = cfg["subset"]
        if MODELSCOPE_TOKEN:
            kwargs["token"] = MODELSCOPE_TOKEN
        try:
            ds = MsDataset.load(**kwargs, use_streaming=True)
        except TypeError:
            ds = MsDataset.load(**kwargs)
        for i, row in enumerate(ds):
            if i < doc_offset:
                continue
            if _pass(row):
                t = _get_text(row)
                if t and (not skip_jp or _no_jp(t)):
                    yield t


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def preflight_check(datasets: list[dict]) -> bool:
    print("\n" + "═" * 60)
    print("  PRE-FLIGHT — vérification accès datasets")
    print("═" * 60)
    all_ok = True

    for cfg in tqdm(datasets, desc="Pre-flight", unit="dataset"):
        name = cfg["name"]
        try:
            gen = stream_dataset(cfg, doc_offset=0)
            doc = next(gen, None)
            if doc is None:
                print(f"  ✗ {name} : stream vide (aucun document retourné)")
                all_ok = False
            else:
                preview = doc[:60].replace("\n", " ")
                print(f"  ✓ {name:<30}  → \"{preview}…\"")
        except Exception as e:
            print(f"  ✗ {name} : {e}")
            all_ok = False

    print("═" * 60)
    if all_ok:
        print("  ✓ Tous les datasets sont accessibles — download autorisé\n")
    else:
        print("  ✗ Certains datasets sont inaccessibles — ABANDON\n")
    return all_ok


# ── Reprise ────────────────────────────────────────────────────────────────────

def _dataset_tokens_on_disk(name: str) -> int:
    """Retourne le nombre de tokens réellement écrits sur disque pour un dataset."""
    out_dir = CHUNKS_DIR / name
    if not out_dir.exists():
        return 0
    bpt = np.dtype(DTYPE).itemsize
    return sum(
        p.stat().st_size // bpt
        for p in out_dir.glob("chunk_*.bin")
        if p.stat().st_size > 0
    )


def load_resume() -> dict:
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            data = json.load(f)
        data.setdefault("failed", [])
        return data
    return {"completed": [], "in_progress": {}, "failed": []}


def _compute_progress_from_disk(name: str, toks_on_disk: int) -> dict:
    """
    Reconstruit un état in_progress cohérent depuis les fichiers sur disque.
    Utilisé quand resume.json est absent, corrompu ou incohérent.
    """
    bpt     = np.dtype(DTYPE).itemsize
    out_dir = CHUNKS_DIR / name
    chunks  = sorted(
        p for p in out_dir.glob("chunk_*.bin")
        if p.stat().st_size > 0
    ) if out_dir.exists() else []

    full_chunks = sum(1 for c in chunks if c.stat().st_size // bpt >= CHUNK_TOKENS)
    chunk_idx   = full_chunks
    chunk_toks  = max(0, toks_on_disk - full_chunks * CHUNK_TOKENS)

    avg_toks   = _AVG_TOKS_PER_DOC.get(name, 500)
    doc_offset = int(toks_on_disk / max(avg_toks, 1))

    return {
        "chunk_idx"  : chunk_idx,
        "chunk_toks" : chunk_toks,
        "doc_offset" : doc_offset,
        "tokens_done": toks_on_disk,
    }


def validate_resume(state: dict) -> dict:
    """
    Auto-repair du resume.json en 3 passes :
    1. Completed → retire les datasets sans données suffisantes sur disque.
    2. In_progress → recalcule depuis le disque si tokens_done est incohérent.
    3. Orphelins → détecte les datasets qui ont des données sur disque mais
       sont absents du resume → les ajoute automatiquement.
    """
    completed   = state.get("completed", [])
    in_progress = state.get("in_progress", {})
    changed     = False

    # Passe 1 : vérification des datasets "completed"
    valid_completed: list[str] = []
    for name in completed:
        cfg = next((d for d in DATASETS if d["name"] == name), None)
        if cfg is None:
            valid_completed.append(name)
            continue
        toks_on_disk = _dataset_tokens_on_disk(name)
        min_required = int(cfg["token_target"] * MIN_COMPLETION_RATIO)
        if toks_on_disk >= min_required:
            valid_completed.append(name)
        else:
            print(f"  ⚠  '{name}' marqué completed mais seulement "
                  f"{toks_on_disk/1e9:.3f}B sur disque "
                  f"(min {min_required/1e9:.3f}B) → remis en attente")
            in_progress.pop(name, None)
            changed = True

    # Passe 2 : vérification des datasets "in_progress"
    for name, prog in list(in_progress.items()):
        toks_on_disk = _dataset_tokens_on_disk(name)
        tokens_saved = prog.get("tokens_done", 0)
        if toks_on_disk < tokens_saved * 0.95:
            new_prog = _compute_progress_from_disk(name, toks_on_disk)
            print(f"  ⚠  '{name}' in_progress incohérent : "
                  f"resume={tokens_saved/1e9:.3f}B  disque={toks_on_disk/1e9:.3f}B "
                  f"→ recalculé depuis disque "
                  f"(chunk={new_prog['chunk_idx']}, doc≈{new_prog['doc_offset']:,})")
            in_progress[name] = new_prog
            changed = True

    # Passe 3 : détection des datasets orphelins
    tracked = set(valid_completed) | set(in_progress.keys())
    for cfg in DATASETS:
        name = cfg["name"]
        if name in tracked:
            continue
        toks_on_disk = _dataset_tokens_on_disk(name)
        if toks_on_disk == 0:
            continue
        min_required = int(cfg["token_target"] * MIN_COMPLETION_RATIO)
        if toks_on_disk >= min_required:
            valid_completed.append(name)
            print(f"  ✓  '{name}' auto-détecté completed "
                  f"({toks_on_disk/1e9:.3f}B ≥ {min_required/1e9:.3f}B)")
        else:
            prog = _compute_progress_from_disk(name, toks_on_disk)
            in_progress[name] = prog
            print(f"  ↩  '{name}' auto-détecté in_progress "
                  f"({toks_on_disk/1e9:.3f}B — chunk={prog['chunk_idx']}, "
                  f"doc≈{prog['doc_offset']:,})")
        changed = True

    state["completed"]   = valid_completed
    state["in_progress"] = in_progress
    if changed:
        save_resume(state)
        print(f"  ✓  resume.json mis à jour automatiquement → {RESUME_FILE}")

    return state


def save_resume(state: dict):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Écriture chunked ───────────────────────────────────────────────────────────

class ChunkWriter:
    """
    Écrit un flux de tokens en fichiers chunk_NNN.bin dans `out_dir`.
    Chaque fichier contient exactement CHUNK_TOKENS tokens, sauf le dernier.
    Le fichier n'est créé sur disque qu'au premier write() (pas de fichiers
    vides en cas de crash avant écriture).
    """

    def __init__(self, out_dir: Path, start_chunk: int = 0, chunk_offset: int = 0):
        self.out_dir    = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_idx  = start_chunk
        self.chunk_toks = chunk_offset
        self.buf: list  = []
        self.buf_size   = 0
        self.f          = None   # ouverture différée au premier write()

    def _chunk_path(self) -> Path:
        return self.out_dir / f"chunk_{self.chunk_idx:03d}.bin"

    def _ensure_open(self):
        if self.f is None:
            mode   = "ab" if self.chunk_toks > 0 else "wb"
            self.f = open(self._chunk_path(), mode)

    def _flush_buf(self):
        if self.buf:
            self._ensure_open()
            np.array(self.buf, dtype=DTYPE).tofile(self.f)
            self.f.flush()
            self.buf      = []
            self.buf_size = 0

    def write(self, tokens: list[int]):
        pos = 0
        while pos < len(tokens):
            space           = CHUNK_TOKENS - self.chunk_toks
            take            = min(space, len(tokens) - pos)
            self.buf.extend(tokens[pos : pos + take])
            self.buf_size  += take
            self.chunk_toks += take
            pos             += take

            if self.buf_size >= WRITE_BUF_TOKS:
                self._flush_buf()

            if self.chunk_toks >= CHUNK_TOKENS:
                self._flush_buf()
                if self.f:
                    self.f.close()
                    self.f = None
                self.chunk_idx  += 1
                self.chunk_toks  = 0

    def close(self) -> tuple[int, int]:
        self._flush_buf()
        if self.f:
            self.f.close()
            self.f = None
        return self.chunk_idx, self.chunk_toks

    @property
    def total_chunks_closed(self) -> int:
        return self.chunk_idx


# ── Pipeline producer-consumer ─────────────────────────────────────────────────

def _tok_producer(cfg: dict, doc_offset: int,
                  raw_q: Queue, stop_evt: threading.Event) -> None:
    """Thread 1 — stream ModelScope/HF → raw_q."""
    try:
        for text in stream_dataset(cfg, doc_offset):
            if stop_evt.is_set():
                break
            raw_q.put(text)
    finally:
        raw_q.put(_SENTINEL)


def _tok_flush_batch(batch: list[str], tok_q: Queue, tokenizer) -> None:
    """Tokenise un batch Rust et pousse les ids dans tok_q."""
    if not batch:
        return
    result = tokenizer(
        batch,
        add_special_tokens    = False,
        truncation            = False,
        return_attention_mask = False,
        return_token_type_ids = False,
    ).input_ids
    for ids in result:
        ids.append(EOS_ID)
        tok_q.put(ids)


def _tok_worker(raw_q: Queue, tok_q: Queue,
                tokenizer, stop_evt: threading.Event) -> None:
    """Thread 2 — raw_q → batch tokenizer Rust (GIL libéré) → tok_q."""
    batch: list[str] = []
    try:
        while True:
            try:
                item = raw_q.get(timeout=2.0)
            except Empty:
                if stop_evt.is_set() and raw_q.empty():
                    break
                continue

            if item is _SENTINEL:
                _tok_flush_batch(batch, tok_q, tokenizer)
                break

            batch.append(item)
            if len(batch) >= TOKENIZER_BATCH_SIZE:
                _tok_flush_batch(batch, tok_q, tokenizer)
                batch = []
    finally:
        tok_q.put(_SENTINEL)


# ── Download d'un dataset ──────────────────────────────────────────────────────

def download_dataset(cfg: dict, tokenizer, state: dict):
    """
    Pipeline 3 threads pour tokeniser un dataset.
    Ne marque 'completed' que si MIN_COMPLETION_RATIO de la cible est atteint.
    """
    name         = cfg["name"]
    token_target = cfg["token_target"]

    prog         = state.get("in_progress", {}).get(name, {})
    start_chunk  = prog.get("chunk_idx",   0)
    chunk_offset = prog.get("chunk_toks",  0)
    doc_offset   = prog.get("doc_offset",  0)
    tokens_done  = prog.get("tokens_done", 0)

    out_dir = CHUNKS_DIR / name
    writer  = ChunkWriter(out_dir, start_chunk, chunk_offset)

    lang_info = ""
    if cfg.get("quality_filter"):
        lang_info = f"  quality ≥ {cfg['quality_filter']}"
    if cfg.get("lang_filter"):
        lang_info += f"  lang={cfg['lang_filter']}"

    print(f"\n{'─'*60}")
    print(f"  Dataset : {name}  [Phase {cfg['phase']}]"
          + ("  [replay P3]" if cfg.get("replay") else ""))
    print(f"  Cible   : {token_target/1e9:.1f}B tokens  →  {out_dir}/" + lang_info)
    print(f"  Mode    : pipeline 3 threads  (batch={TOKENIZER_BATCH_SIZE})")
    if doc_offset:
        print(f"  Reprise : doc #{doc_offset:,}  chunk #{start_chunk:03d}  "
              f"({tokens_done/1e9:.3f}B tokens déjà écrits)")
    print(f"{'─'*60}")

    raw_q    = Queue(maxsize=RAW_Q_MAX)
    tok_q    = Queue(maxsize=TOK_Q_MAX)
    stop_evt = threading.Event()

    t_prod = threading.Thread(
        target=_tok_producer, daemon=True, name=f"producer-{name}",
        args=(cfg, doc_offset, raw_q, stop_evt),
    )
    t_tok = threading.Thread(
        target=_tok_worker, daemon=True, name=f"tokenizer-{name}",
        args=(raw_q, tok_q, tokenizer, stop_evt),
    )
    t_prod.start()
    t_tok.start()

    pbar = tqdm(
        total=token_target, initial=tokens_done,
        unit="tok", unit_scale=True, desc=name, dynamic_ncols=True,
    )

    doc_count       = doc_offset
    save_every      = 10_000
    last_save       = doc_count
    toks_since_save = 0
    t0_speed        = time.time()

    try:
        while True:
            try:
                ids = tok_q.get(timeout=30.0)
            except Empty:
                if not t_tok.is_alive():
                    break
                continue

            if ids is _SENTINEL:
                break

            writer.write(ids)
            n                = len(ids)
            tokens_done     += n
            toks_since_save += n
            doc_count       += 1

            pbar.update(n)
            pbar.set_postfix(
                docs  = f"{doc_count:,}",
                chunk = f"{writer.total_chunks_closed}",
                Mtps  = f"{toks_since_save / max(time.time() - t0_speed, 1) / 1e6:.2f}",
                Qraw  = raw_q.qsize(),
                Qtok  = tok_q.qsize(),
            )

            if tokens_done >= token_target:
                stop_evt.set()
                break

            if doc_count - last_save >= save_every:
                state.setdefault("in_progress", {})[name] = {
                    "chunk_idx"  : writer.chunk_idx,
                    "chunk_toks" : writer.chunk_toks,
                    "doc_offset" : doc_count,
                    "tokens_done": tokens_done,
                }
                save_resume(state)
                last_save       = doc_count
                toks_since_save = 0
                t0_speed        = time.time()

    finally:
        stop_evt.set()
        pbar.close()

    t_prod.join(timeout=15)
    t_tok.join(timeout=15)

    final_chunk, final_toks = writer.close()
    n_chunks = final_chunk + (1 if final_toks > 0 else 0)
    print(f"  ✓ {name} : {tokens_done/1e9:.3f}B tokens  "
          f"{doc_count:,} docs  {n_chunks} chunks")

    min_required = int(token_target * MIN_COMPLETION_RATIO)
    if tokens_done >= min_required:
        state.setdefault("completed", []).append(name)
        state.get("in_progress", {}).pop(name, None)
        if name in state.get("failed", []):
            state["failed"].remove(name)
        print(f"  ✓ {name} marqué 'completed' ({tokens_done/1e9:.3f}B ≥ {min_required/1e9:.3f}B)")
    else:
        state.setdefault("in_progress", {})[name] = {
            "chunk_idx"  : final_chunk,
            "chunk_toks" : final_toks,
            "doc_offset" : doc_count,
            "tokens_done": tokens_done,
        }
        print(f"  ⚠  {name} incomplet : {tokens_done/1e9:.3f}B / {token_target/1e9:.1f}B "
              f"→ sera repris au prochain lancement")

    save_resume(state)
    return tokens_done


# ── Scan d'un fichier chunk ────────────────────────────────────────────────────

def scan_chunk(path: Path) -> list[tuple[int, int]]:
    """Retourne [(start, length), ...] pour chaque document du chunk."""
    if path.stat().st_size == 0:
        return []
    data          = np.fromfile(path, dtype=DTYPE)
    eos_positions = np.where(data == EOS_ID)[0]
    docs  = []
    start = 0
    for pos in eos_positions:
        length = int(pos) - start + 1
        if length > 1:
            docs.append((start, length))
        start = int(pos) + 1
    return docs


def scan_dataset_chunks(name: str) -> list[tuple[Path, int, int]]:
    out_dir  = CHUNKS_DIR / name
    chunks   = sorted(p for p in out_dir.glob("chunk_*.bin") if p.stat().st_size > 0)
    all_docs : list[tuple[Path, int, int]] = []
    for chunk_path in tqdm(chunks, desc=f"  scan {name}", unit="chunk", leave=False):
        for start, length in scan_chunk(chunk_path):
            all_docs.append((chunk_path, start, length))
    return all_docs


# ── Écriture finale ────────────────────────────────────────────────────────────

def _read_doc(chunk_path: Path, offset: int, length: int) -> np.ndarray:
    bpt = np.dtype(DTYPE).itemsize
    with open(chunk_path, "rb") as f:
        f.seek(offset * bpt)
        return np.frombuffer(f.read(length * bpt), dtype=DTYPE).copy()


def _split_path(idx: int) -> Path:
    return Path(f"{OUT_PREFIX}_{idx:03d}.bin")


def upload_split_to_hf(file_path: Path, file_idx: int):
    if not HF_TOKEN:
        print(f"  ⚠  HF_TOKEN absent — upload ignoré pour {file_path.name}")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        print(f"  ↑ Upload {file_path.name} → {HF_DATASET_REPO} …")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo   =file_path.name,
            repo_id        =HF_DATASET_REPO,
            repo_type      ="dataset",
            commit_message =f"pretrain split {file_idx:03d} ({file_path.stat().st_size/1e9:.1f} GB)",
        )
        print(f"  ✓ Upload OK → {HF_DATASET_REPO}/{file_path.name}")
    except ImportError:
        print("  ✗ huggingface_hub non installé : pip install huggingface_hub")
    except Exception as e:
        print(f"  ✗ Upload échoué pour {file_path.name} : {e}")


class _SplitWriter:
    def __init__(self, start_idx: int = 0):
        self.file_idx = start_idx
        self.toks_in  = 0
        self.total    = 0
        self.buf: list[np.ndarray] = []
        self.buf_size = 0
        self._open()

    def _open(self):
        p = _split_path(self.file_idx)
        print(f"\n  ✎ Ouverture {p.name}  (fichier {self.file_idx+1})")
        self.f = open(p, "wb")

    def _flush(self):
        if self.buf:
            np.concatenate(self.buf).tofile(self.f)
            self.f.flush()
            self.buf      = []
            self.buf_size = 0

    def write_doc(self, arr: np.ndarray):
        remaining = arr
        while len(remaining):
            space = SPLIT_TOKENS - self.toks_in
            take  = min(space, len(remaining))
            self.buf.append(remaining[:take])
            self.buf_size += take
            self.toks_in  += take
            self.total    += take
            remaining      = remaining[take:]

            if self.buf_size >= WRITE_BUF_TOKS:
                self._flush()

            if self.toks_in >= SPLIT_TOKENS:
                self._flush()
                self.f.close()
                p = _split_path(self.file_idx)
                print(f"  ✓ {p.name} complet : {self.toks_in/1e9:.1f}B tokens  "
                      f"({p.stat().st_size/1e9:.1f} GB)")
                upload_split_to_hf(p, self.file_idx)
                self.file_idx += 1
                self.toks_in  = 0
                self._open()

    def close(self) -> int:
        self._flush()
        self.f.close()
        p = _split_path(self.file_idx)
        if p.stat().st_size > 0:
            print(f"  ✓ {p.name} (dernier) : {self.toks_in/1e9:.1f}B tokens  "
                  f"({p.stat().st_size/1e9:.1f} GB)")
            upload_split_to_hf(p, self.file_idx)
            return self.file_idx + 1
        else:
            p.unlink(missing_ok=True)
            return self.file_idx


def _write_docs(docs: list[tuple[Path, int, int]], writer: "_SplitWriter", desc: str = ""):
    pbar = tqdm(docs, desc=desc, unit="doc", dynamic_ncols=True, leave=False)
    for chunk_path, offset, length in pbar:
        arr = _read_doc(chunk_path, offset, length)
        writer.write_doc(arr)
    pbar.close()


def _interleave_replay(
    p2_docs: list[tuple[Path, int, int]],
    replay_docs: list[tuple[Path, int, int]],
    ratio_start: float = REPLAY_RATIO_START,
    ratio_end: float   = REPLAY_RATIO_END,
) -> list[tuple[Path, int, int]]:
    result: list[tuple[Path, int, int]] = []
    r_idx = 0
    budget = 0.0
    n_p2 = len(p2_docs)
    n_r  = len(replay_docs)

    for i, doc in enumerate(p2_docs):
        frac   = i / max(n_p2 - 1, 1)
        ratio  = ratio_start + (ratio_end - ratio_start) * frac
        budget += ratio / max(1.0 - ratio, 1e-6)
        result.append(doc)
        while budget >= 1.0 and r_idx < n_r:
            result.append(replay_docs[r_idx])
            r_idx  += 1
            budget -= 1.0

    while r_idx < n_r:
        result.append(replay_docs[r_idx])
        r_idx += 1

    return result


# ── Assembly ───────────────────────────────────────────────────────────────────

def assemble(state: dict):
    print("\n" + "═" * 60)
    print("  ASSEMBLY — curriculum 2 phases  →  pretrain_data_NNN.bin")
    print("═" * 60)

    bpt = np.dtype(DTYPE).itemsize
    total_tokens_on_disk = 0
    for d in DATASETS:
        out_dir = CHUNKS_DIR / d["name"]
        for p in out_dir.glob("chunk_*.bin"):
            total_tokens_on_disk += p.stat().st_size // bpt

    p1_end_tok   = int(total_tokens_on_disk * P1_END_FRAC)
    p3_start_tok = int(total_tokens_on_disk * P3_START_FRAC)
    print(f"\n  Tokens totaux sur disque : {total_tokens_on_disk/1e9:.1f}B")
    print(f"  Phase 1 : 0 → {p1_end_tok/1e9:.1f}B  ({P1_END_FRAC*100:.0f}%)")
    print(f"  Phase 2 : {p1_end_tok/1e9:.1f}B → {p3_start_tok/1e9:.1f}B  "
          f"({(P3_START_FRAC-P1_END_FRAC)*100:.0f}%)")
    print(f"  Phase 3 : {p3_start_tok/1e9:.1f}B → fin  ({(1-P3_START_FRAC)*100:.0f}%)")

    print("\n[1/5] Scan CosmopediaV2…")
    cosmo_docs = scan_dataset_chunks("cosmopedia_v2")
    print(f"      {len(cosmo_docs):,} documents")

    cosmo_toks = sum(l for _, _, l in cosmo_docs)
    if cosmo_toks <= p1_end_tok:
        p1_docs       = cosmo_docs
        cosmo_p2_docs = []
    else:
        acc = split = 0
        for i, (_, _, l) in enumerate(cosmo_docs):
            if acc + l > p1_end_tok:
                split = i
                break
            acc += l
        p1_docs       = cosmo_docs[:split]
        cosmo_p2_docs = cosmo_docs[split:]

    print(f"      → Phase 1 : {len(p1_docs):,} docs")
    print(f"      → Phase 2 : {len(cosmo_p2_docs):,} docs (excédent Cosmo)")

    print("\n[2/5] Scan datasets Phase 2…")
    p2_names = [d["name"] for d in DATASETS if d["name"] != "cosmopedia_v2"]
    all_p2_docs: list[tuple[Path, int, int]] = list(cosmo_p2_docs)

    for name in p2_names:
        ds_docs = scan_dataset_chunks(name)
        print(f"      {name:<30} {len(ds_docs):>8,} docs")
        all_p2_docs.extend(ds_docs)

    print(f"\n      Total Phase 2 brut : {len(all_p2_docs):,} documents")

    print("\n[3/5] Shuffle Phase 2…")
    rng = random.Random(42)
    rng.shuffle(all_p2_docs)
    print(f"      ✓ {len(all_p2_docs):,} docs shufflés (seed=42)")

    p1_token_count = sum(l for _, _, l in p1_docs)
    acc = p1_token_count
    split = len(all_p2_docs)
    for i, (_, _, l) in enumerate(all_p2_docs):
        if acc >= p3_start_tok:
            split = i
            break
        acc += l

    p2_docs      = all_p2_docs[:split]
    p3_base_docs = all_p2_docs[split:]
    print(f"      Phase 2 pure  : {len(p2_docs):,} docs")
    print(f"      Phase 3 base  : {len(p3_base_docs):,} docs")

    print("\n[4/5] Préparation replay EN Phase 3…")
    p3_base_toks  = sum(l for _, _, l in p3_base_docs)
    ratio_moy     = (REPLAY_RATIO_START + REPLAY_RATIO_END) / 2
    replay_budget = int(p3_base_toks * ratio_moy / max(1.0 - ratio_moy, 1e-6))
    print(f"      Tokens Phase 3 base   : {p3_base_toks/1e9:.2f}B")
    print(f"      Budget replay cible   : {replay_budget/1e9:.2f}B tokens "
          f"(ratio moy {ratio_moy*100:.0f}%)")

    replay_docs: list[tuple[Path, int, int]] = []
    for name in REPLAY_DATASETS:
        rd = scan_dataset_chunks(name)
        rng.shuffle(rd)
        per_ds_budget = replay_budget // len(REPLAY_DATASETS)
        acc_r = 0
        taken = []
        for doc in rd:
            if acc_r >= per_ds_budget:
                break
            taken.append(doc)
            acc_r += doc[2]
        replay_docs.extend(taken)
        print(f"      {name:<30} {len(taken):>8,} docs  ({acc_r/1e9:.2f}B tokens)")

    rng.shuffle(replay_docs)
    p3_docs = _interleave_replay(p3_base_docs, replay_docs)
    print(f"      ✓ Phase 3 finale : {len(p3_docs):,} docs "
          f"({len(p3_base_docs):,} P2 + {len(replay_docs):,} replay)")

    total_docs = len(p1_docs) + len(p2_docs) + len(p3_docs)
    n_splits   = max(1, (sum(l for _, _, l in p1_docs + p2_docs + p3_docs)
                         + SPLIT_TOKENS - 1) // SPLIT_TOKENS)
    print(f"\n[5/5] Écriture → {OUT_PREFIX}_NNN.bin  "
          f"({total_docs:,} docs  ~{n_splits} fichiers)")
    print(f"      Chaque fichier sera uploadé sur {HF_DATASET_REPO} dès sa clôture.")

    writer = _SplitWriter(start_idx=0)

    print(f"\n  ━━ Phase 1 : Cosmopedia pur ({len(p1_docs):,} docs) ━━")
    _write_docs(p1_docs, writer, desc="  Phase 1")

    print(f"\n  ━━ Phase 2 : Mix shufflé ({len(p2_docs):,} docs) ━━")
    _write_docs(p2_docs, writer, desc="  Phase 2")

    print(f"\n  ━━ Phase 3 : Replay EN progressif ({len(p3_docs):,} docs) ━━")
    _write_docs(p3_docs, writer, desc="  Phase 3")

    n_files = writer.close()
    print(f"\n  ✓ Assembly terminé : {writer.total/1e9:.3f}B tokens  "
          f"{n_files} fichier(s)  ({writer.total * np.dtype(DTYPE).itemsize / 1e9:.0f} GB total)")


# ── Résumé ─────────────────────────────────────────────────────────────────────

def print_summary():
    bpt          = np.dtype(DTYPE).itemsize
    total_target = sum(d["token_target"] for d in DATASETS)
    print(f"\n{'━'*60}")
    print(f"  RÉSUMÉ — mix 50B tokens")
    print(f"{'━'*60}")
    print(f"  {'Nom':<28} {'Ph':>2}  {'Cible':>6}  {'Disque':>8}  {'Chunks'}")
    print(f"  {'─'*56}")
    for d in DATASETS:
        out_dir = CHUNKS_DIR / d["name"]
        chunks  = [p for p in out_dir.glob("chunk_*.bin") if p.stat().st_size > 0] if out_dir.exists() else []
        actual  = sum(p.stat().st_size for p in chunks) // bpt if chunks else 0
        replay  = " ♻" if d.get("replay") else ""
        pct     = actual / d["token_target"] * 100
        print(f"  {d['name']:<28} {d['phase']:>2}  "
              f"{d['token_target']/1e9:>5.1f}B  "
              f"{actual/1e9:>7.2f}B ({pct:>5.1f}%)  "
              f"{len(chunks)} chunks{replay}")
    print(f"  {'─'*56}")
    print(f"  {'TOTAL':<28}     {total_target/1e9:>5.1f}B")
    print(f"{'━'*60}")
    splits = sorted(Path(".").glob(f"{OUT_PREFIX}_*.bin"))
    if splits:
        total_split_toks = sum(p.stat().st_size for p in splits) // bpt
        total_split_gb   = sum(p.stat().st_size for p in splits) / 1e9
        print(f"\n  Fichiers de pretrain ({len(splits)}) :")
        for p in splits:
            n = p.stat().st_size // bpt
            print(f"    {p.name}  {n/1e9:.1f}B tokens  ({p.stat().st_size/1e9:.1f} GB)")
        print(f"  Total : {total_split_toks/1e9:.1f}B tokens  ({total_split_gb:.0f} GB)")
    print(f"{'━'*60}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global TOKENIZER_BATCH_SIZE

    parser = argparse.ArgumentParser(
        description="Tokenisation + assembly curriculum 50B tokens"
    )
    parser.add_argument("--reset",         action="store_true",
                        help="Repart de zéro (supprime resume.json)")
    parser.add_argument("--skip-assembly", action="store_true",
                        help="Tokenise uniquement, sans assembly final")
    parser.add_argument("--only-assembly", action="store_true",
                        help="Saute le download, lance seulement l'assembly")
    parser.add_argument("--no-preflight",  action="store_true",
                        help="Saute le pre-flight check")
    parser.add_argument("--batch-size",    type=int, default=TOKENIZER_BATCH_SIZE,
                        metavar="N",
                        help=f"Docs par batch tokenizer (défaut {TOKENIZER_BATCH_SIZE})")
    args = parser.parse_args()

    TOKENIZER_BATCH_SIZE = args.batch_size

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        if RESUME_FILE.exists():
            RESUME_FILE.unlink()
            print("⚠  Fichier de reprise supprimé — repartir de zéro.")

    if not args.only_assembly and not args.no_preflight:
        ok = preflight_check(DATASETS)
        if not ok:
            sys.exit(1)

    state = (load_resume() if not args.reset
             else {"completed": [], "in_progress": {}, "failed": []})

    tokenizer = load_tokenizer()

    if not args.reset:
        print("\n  Auto-repair resume.json…")
        state = validate_resume(state)
        print("  ✓ resume.json cohérent avec le disque")

    random.seed(42)

    if not args.only_assembly:
        completed   = state.get("completed", [])
        prev_failed = state.get("failed", [])
        pending     = [d for d in DATASETS if d["name"] not in completed]

        if prev_failed:
            print(f"\n  ⚠  {len(prev_failed)} dataset(s) avaient échoué :")
            for n in prev_failed:
                print(f"     - {n}")

        if not pending:
            print("  ✓ Tous les datasets sont déjà complétés.")
        else:
            print(f"  → {len(pending)} dataset(s) à télécharger : "
                  f"{', '.join(d['name'] for d in pending)}")

        for cfg in pending:
            try:
                download_dataset(cfg, tokenizer, state)
            except Exception as exc:
                name = cfg["name"]
                print(f"\n  ✗ {name} a échoué : {exc}")
                failed_list = state.setdefault("failed", [])
                if name not in failed_list:
                    failed_list.append(name)
                state.get("in_progress", {}).pop(name, None)
                save_resume(state)

        n_failed = len(state.get("failed", []))
        if n_failed:
            print(f"\n  ⚠  {n_failed} dataset(s) ont échoué. Relance : python tokens.py")
        else:
            print("\n  ✓ Tokenisation terminée.")

    if not args.skip_assembly:
        assemble(state)

    print_summary()
    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
