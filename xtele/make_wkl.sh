#!/bin/bash
set -e
python3 setup.py build_ext --inplace

sudo find . -name "*.so" -exec strip --strip-all {} \;

python3 -m build --wheel
find . \( -name "*.c" -o -name "*.cpp" \) -print
find . \( -name "*.c" -o -name "*.cpp" \) -delete

find . \( -name "*.o" -o -name "*.so" \) -print
find . \( -name "*.o" -o -name "*.so" \) -delete
