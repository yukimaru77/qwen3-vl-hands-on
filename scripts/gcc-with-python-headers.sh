#!/usr/bin/env bash
exec /usr/bin/gcc \
  -I/project/local/python312deb/extract/usr/include \
  -I/project/local/python312deb/extract/usr/include/python3.12 \
  "$@"
