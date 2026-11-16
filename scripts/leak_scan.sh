#!/usr/bin/env bash
# Leak scan: grep every tracked file (and every tracked path) for the terms of a deny-list.
#
#   scripts/leak_scan.sh <denylist-file>
#
# The deny-list lives outside the repository and is never committed. It holds one PCRE per line;
# blank lines and lines starting with '#' are ignored. Matching is case-insensitive, and a
# pattern that starts (ends) on a letter only matches where the character next to it is not a
# letter, so a short term never hits inside a longer word but does hit inside a snake_case or
# numbered name ("term_01", "term2"). A pattern that starts (ends) on a digit, '_' or a group
# only matches where its neighbour is not a letter or digit (an id never hits inside a longer
# number). Patterns that start or end on another character (a path such as "/opt/", an escaped
# dot) carry their own anchoring.
#
# Prints one "path:line: [pattern] text" row per hit and exits 0 only when nothing matches
# (1 on hits, 2 on a usage error).
set -u

list="${1:-}"
if [ -z "$list" ] || [ ! -r "$list" ]; then
    echo "usage: scripts/leak_scan.sh <denylist-file> (readable)" >&2
    exit 2
fi
root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "leak_scan: not in a git repo" >&2; exit 2; }
cd "$root" || exit 2

GREP=/usr/bin/grep
[ -x "$GREP" ] || GREP="$(command -v grep)"

files=()
while IFS= read -r -d '' f; do
    [ -f "$f" ] && files+=("$f")
done < <(git ls-files -z)
if [ "${#files[@]}" -eq 0 ]; then
    echo "leak_scan: no tracked files" >&2
    exit 2
fi

hits=0
patterns=0
while IFS= read -r raw || [ -n "$raw" ]; do
    pat="${raw%$'\r'}"
    case "$pat" in ''|'#'*) continue ;; esac
    patterns=$((patterns + 1))
    pre="" post=""
    case "${pat:0:1}" in
        [A-Za-z]) pre='(?<![A-Za-z])' ;;
        [0-9_\[\(]) pre='(?<![A-Za-z0-9])' ;;
    esac
    case "${pat: -1}" in
        [A-Za-z]) post='(?![A-Za-z])' ;;
        [0-9_\]\)+*}?]) post='(?![A-Za-z0-9])' ;;
    esac
    re="${pre}(?:${pat})${post}"
    out="$("$GREP" -nHIiP -e "$re" -- "${files[@]}" 2>/dev/null)"
    if [ -n "$out" ]; then
        while IFS= read -r line; do
            echo "${line%%:*}:$(echo "$line" | cut -d: -f2): [$pat] $(echo "$line" | cut -d: -f3- | cut -c1-200)"
            hits=$((hits + 1))
        done <<< "$out"
    fi
    pout="$(git ls-files | "$GREP" -iP -e "$re" 2>/dev/null)"
    if [ -n "$pout" ]; then
        while IFS= read -r p; do
            echo "PATH ${p}: [$pat]"
            hits=$((hits + 1))
        done <<< "$pout"
    fi
done < "$list"

echo "leak_scan: ${#files[@]} tracked files, ${patterns} patterns, ${hits} hit(s)"
[ "$hits" -eq 0 ]
