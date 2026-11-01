#!/usr/bin/env bash
#
# Fetch the S6E11 competition data into data/.
#
#   ./scripts/fetch_data.sh          # no-op if the files are already present
#   ./scripts/fetch_data.sh --force  # re-download and overwrite
#
# Requires the Kaggle CLI (`pip install kaggle`) and credentials, on an account
# that has joined the competition. Current CLI versions accept any of:
#   kaggle auth login                 OAuth, no token file to manage
#   export KAGGLE_API_TOKEN=...       environment variable
#   ~/.kaggle/access_token            token file
#   ~/.kaggle/kaggle.json             legacy format, older CLIs
# This script never reads, prints or copies any of them.
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

# Credential discovery mirrors what the CLI itself accepts. Presence is all
# that is checked; the contents are never read by this script.
KAGGLE_DIR="${KAGGLE_CONFIG_DIR:-${HOME}/.kaggle}"
if [[ -z "${KAGGLE_API_TOKEN:-}" \
   && -z "${KAGGLE_USERNAME:-}" \
   && ! -f "${KAGGLE_DIR}/access_token" \
   && ! -f "${KAGGLE_DIR}/kaggle.json" \
   && ! -d "${HOME}/.cache/kaggle" ]]; then
  cat >&2 <<MSG
error: no Kaggle credentials found in ${KAGGLE_DIR}, the environment, or the
OAuth cache.

Easiest — OAuth, nothing to store:

  kaggle auth login

Or generate a token at https://www.kaggle.com/settings/api and either:

  export KAGGLE_API_TOKEN=...          # this shell only
  # or write it to ${KAGGLE_DIR}/access_token , then:
  chmod 600 ${KAGGLE_DIR}/access_token

(~/.kaggle/kaggle.json is the legacy format and still works on older CLIs.)
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
