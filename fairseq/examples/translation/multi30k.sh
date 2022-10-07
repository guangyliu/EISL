#!/usr/bin/env bash

DATA_DIR=${DATA_DIR:-multi30k_dataset}
BPE_DIR=${BPE_DIR:-subword-nmt}

SRC=${SRC:-en}
TGT=${TGT:-de}
PREP=${PREP:-wmt18.multi30k."$SRC"-"$TGT"}

DATA_REPO='https://github.com/multi30k/dataset.git'
TOK_DATA_DIR="$DATA_DIR/data/task1/tok"
BPE_REPO='https://github.com/rsennrich/subword-nmt.git'
BPE_CODE="$PREP/code"
BPE_TOKENS=10000

if [ ! -d "$BPE_DIR" ]; then
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone "$BPE_REPO" "$BPE_DIR"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo 'Cloning WMT 2018 Multi30k Data Repository'
    git clone "$DATA_REPO" "$DATA_DIR"
fi

mkdir -p "$PREP"

echo "learn_bpe.py on $TOK_DATA_DIR/train.lc.norm.tok.{$SRC,$TGT} …"
cat "$TOK_DATA_DIR/train.lc.norm.tok."{"$SRC","$TGT"} | \
    python "$BPE_DIR/subword_nmt/learn_bpe.py" -s "$BPE_TOKENS" >"$BPE_CODE"

for l in "$SRC" "$TGT"; do
    for f in "$TOK_DATA_DIR"/*"$l"; do
        base=$(basename "$f")
        echo "apply_bpe.py to $f …"
        python "$BPE_DIR/subword_nmt/apply_bpe.py" -c "$BPE_CODE" <"$f" >"$PREP/$base"
    done
done

