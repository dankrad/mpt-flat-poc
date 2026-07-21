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
