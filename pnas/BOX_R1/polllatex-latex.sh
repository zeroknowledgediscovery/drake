#!/usr/bin/env bash

set -e

usage() {
    echo "Usage: $(basename "$0") [-e engine] texfile"
    echo
    echo "Options:"
    echo "  -e engine    LaTeX engine to use (pdflatex, lualatex, xelatex)"
    echo "                Default: pdflatex"
    exit 1
}

engine="pdflatex"

while getopts ":e:h" opt; do
    case ${opt} in
        e )
            engine="$OPTARG"
            ;;
        h )
            usage
            ;;
        \? )
            echo "Invalid option: -$OPTARG"
            usage
            ;;
    esac
done

shift $((OPTIND - 1))

if [ -z "$1" ]; then
    usage
fi

texfile="$1"

if [ ! -f "$texfile" ] && [ ! -f "$texfile.tex" ]; then
    echo "Error: TeX file not found: $texfile"
    exit 1
fi

echo "Using engine: $engine"
echo "Watching for changes..."

latexmk \
    -pdf \
    -pvc \
    -interaction=nonstopmode \
    -synctex=1 \
    -pdflatex="$engine --shell-escape %O %S" \
    "$texfile"
