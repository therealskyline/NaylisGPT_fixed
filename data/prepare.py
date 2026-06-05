import argparse
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"
TOKENIZER_LOCAL      = Path("tokenizer")


def load_tokenizer(tokenizer_id: str) -> AutoTokenizer:
    src = str(TOKENIZER_LOCAL) if TOKENIZER_LOCAL.exists() else tokenizer_id
    print(f"Tokenizer : {src}")
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    think_tokens = ["<think>", "</think>"]
    missing = [t for t in think_tokens
               if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        print(f"  + Tokens ajoutés : {missing}")
        TOKENIZER_LOCAL.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(TOKENIZER_LOCAL))
        print(f"  ✓ Tokenizer sauvegardé → {TOKENIZER_LOCAL}/")
    else:
        print(f"  ✓ Tokens <think></think> déjà présents")

    return tok


def resolve_dtype(dtype_arg: str, vocab_size: int) -> np.dtype:
    if dtype_arg == "uint16":
        return np.uint16
    if dtype_arg == "uint32":
        return np.uint32
    return np.uint16 if vocab_size <= 65_535 else np.uint32


def main():
    parser = argparse.ArgumentParser(description="Prépare pretrain_data.bin")
    parser.add_argument("--dataset",    type=str, required=True)
    parser.add_argument("--output",     type=str, default="./pretrain_data.bin")
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--column",     type=str, default="text")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--hf-token",   type=str, default=None)
    parser.add_argument("--tokenizer",  type=str, default=DEFAULT_TOKENIZER_ID,
                        help="HuggingFace tokenizer ID ou chemin local")
    parser.add_argument("--dtype",      type=str, default="auto",
                        choices=["auto", "uint16", "uint32"],
                        help="dtype du fichier .bin (auto = déduit du vocab)")
    args = parser.parse_args()

    tokenizer  = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    DTYPE      = resolve_dtype(args.dtype, vocab_size)
    eos        = tokenizer.eos_token_id

    print(f"  vocab={vocab_size:,}  eos={eos}  dtype={DTYPE.__name__}")
    print(f"\nDataset : {args.dataset}  split={args.split}")
    ds = load_dataset(args.dataset, split=args.split,
                      token=args.hf_token, streaming=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    count = 0
    for sample in tqdm(ds, desc="Tokenisation"):
        if args.limit and count >= args.limit:
            break
        ids = tokenizer.encode(sample[args.column], add_special_tokens=False)
        ids.append(eos)
        all_tokens.extend(ids)
        count += 1

    tokens = np.array(all_tokens, dtype=DTYPE)
    tokens.tofile(output)
    print(f"\nSauvegardé : {output}  "
          f"({len(tokens)/1e9:.3f}B tokens  {output.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
