#!/usr/bin/env bash
#
# Fetch the S6E11 competition data into data/.
#
#   ./scripts/fetch_data.sh          # no-op if the files are already present
#   ./scripts/fetch_data.sh --force  # re-download and overwrite
#
# Requires the Kaggle CLI (`pip install kaggle`) and an API token at
# ~/.kaggle/kaggle.json, on an account that has joined the competition.
# This script never reads, prints or copies the token.
#
set -euo pipefail

COMPETITION="playground-series-s6e11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
EXPECTED=(train.csv test.csv sample_submission.csv)

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

have_all() {
  for f in "${EXPECTED[@]}"; do
    [[ -s "${DATA_DIR}/${f}" ]] || return 1
  done
  return 0
}

if [[ $FORCE -eq 0 ]] && have_all; then
  echo "Data already present in data/ — nothing to do (use --force to re-download)."
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  cat >&2 <<'MSG'
error: the `kaggle` CLI is not on PATH.

  python3 -m venv .venv && source .venv/bin/activate
  pip install kaggle

There is no Homebrew formula for it. If `pip install` reports
"externally-managed-environment", use the venv above or `pipx install kaggle`.

Alternatively download train.csv, test.csv and sample_submission.csv by hand
from https://www.kaggle.com/competitions/playground-series-s6e11/data
and drop them into data/.
MSG
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
  cat >&2 <<'MSG'
error: no Kaggle credentials found.

Create a token at https://www.kaggle.com/settings (API -> Create New Token),
move the downloaded kaggle.json to ~/.kaggle/kaggle.json, then:

  chmod 600 ~/.kaggle/kaggle.json
MSG
  exit 1
fi

mkdir -p "${DATA_DIR}"
echo "Downloading ${COMPETITION} into data/ ..."

if ! kaggle competitions download -c "${COMPETITION}" -p "${DATA_DIR}"; then
  cat >&2 <<MSG

The download failed. A 403 usually means the competition rules have not been
accepted on the account this token belongs to — open
https://www.kaggle.com/competitions/${COMPETITION} , click Join Competition,
and try again.
MSG
  exit 1
fi

ZIP="${DATA_DIR}/${COMPETITION}.zip"
if [[ -f "${ZIP}" ]]; then
  unzip -o -q "${ZIP}" -d "${DATA_DIR}"
  rm -f "${ZIP}"
fi

missing=()
for f in "${EXPECTED[@]}"; do
  [[ -s "${DATA_DIR}/${f}" ]] || missing+=("${f}")
done
if (( ${#missing[@]} )); then
  echo "error: expected file(s) still missing after download: ${missing[*]}" >&2
  exit 1
fi

echo
for f in "${EXPECTED[@]}"; do
  rows=$(($(wc -l < "${DATA_DIR}/${f}") - 1))
  printf '  %-24s %8d rows\n' "${f}" "${rows}"
done
echo
echo "Done. These files are gitignored — do not commit them."
