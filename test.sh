#! /bin/bash

tegrastats |  while IFS= read -r line; do printf '%s %s\n' "$(date '+%s%N')" "$line"; done