#!/usr/bin/env bash
# Export reth's HashedAccounts + HashedStorages (the secure-trie leaves at the
# synced block) to TSV, for reconstructing the state root in our engine.
#   accounts.tsv:  keccak(addr) \t nonce \t balance_hex \t code_hash|null
#   storages.tsv:  keccak(addr) \t keccak(slot) \t value_hex
#
# reth db list buffers `--len` rows in RAM (~545 B/row) and jq loads the chunk's
# JSON, so we page in 20M-row chunks (~20 GB peak, safe under this box's ~52 GB).
# --skip re-walks from the start each chunk (O(skip)); at 20M chunks that's a few
# hundred M cursor steps total — minutes.
set -euo pipefail
DD="${1:?usage: reth-export.sh <reth-datadir> <out-dir>}"
OUT="${2:?usage: reth-export.sh <reth-datadir> <out-dir>}"
RD=(reth db --datadir "$DD" --log.stdout.filter error list)
CHUNK=20000000

dump() { # <table> <jq-filter> <outfile>
  local table=$1 filter=$2 out=$3
  : > "$out"
  local skip=0 before after added
  while :; do
    before=$(wc -l < "$out")
    echo "  $table skip=$skip len=$CHUNK -> $out" >&2
    "${RD[@]}" "$table" --json --skip "$skip" --len "$CHUNK" 2>/dev/null \
      | jq -rc "$filter" >> "$out"
    after=$(wc -l < "$out")
    added=$((after - before))
    # A short (or empty) chunk means we hit the end of the table.
    [ "$added" -lt "$CHUNK" ] && break
    skip=$((skip + CHUNK))
  done
}

# Record which block the hashed tables actually reflect — from the datadir's own
# stage checkpoint — plus that block's header stateRoot. Passing an assumed block's
# root to the loader once cost a full 400M-account load: the datadir had synced
# past the assumed block, the export was faithfully at the newer state, and the
# (correct) reconstructed root was compared against the older block's root.
BLOCK=$(reth db --datadir "$DD" --log.stdout.filter error list StageCheckpoints --json 2>/dev/null \
  | jq -r '.[] | select(.[0] == "AccountHashing") | .[1].block_number')
ROOT=$(reth db --datadir "$DD" --log.stdout.filter error get static-file headers "$BLOCK" 2>/dev/null \
  | jq -r '.stateRoot // empty' || true)
if [ -z "$ROOT" ]; then
  # Older reth prints non-JSON around the value; scrape the field.
  ROOT=$(reth db --datadir "$DD" --log.stdout.filter error get static-file headers "$BLOCK" 2>/dev/null \
    | grep -oE '"stateRoot": *"0x[0-9a-f]{64}"' | grep -oE '0x[0-9a-f]{64}')
fi
echo "hashed state is at block $BLOCK, stateRoot $ROOT" >&2
printf '%s %s\n' "$BLOCK" "$ROOT" > "$OUT/block-$BLOCK.meta"

echo "exporting accounts..." >&2
dump HashedAccounts \
  '.[] | [.[0], (.[1].nonce|tostring), .[1].balance, (.[1].bytecode_hash // "null")] | @tsv' \
  "$OUT/accounts.tsv"

echo "exporting storages..." >&2
dump HashedStorages \
  '.[] | [.[0], .[1].key, .[1].value] | @tsv' \
  "$OUT/storages.tsv"

echo "done. line counts:" >&2
wc -l "$OUT/accounts.tsv" "$OUT/storages.tsv" >&2
