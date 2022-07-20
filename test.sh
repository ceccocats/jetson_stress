#! /bin/bash

sudo nvpmodel -m8
sudo jetson_clocks --fan
sudo nvpmodel -q

tegrastats |  while IFS= read -r line; do printf '%s %s\n' "$(date '+%s%N')" "$line"; done | tee temps.txt &
./test

jobs -p | xargs -r kill
