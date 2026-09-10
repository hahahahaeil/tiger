#!/usr/bin/env bash
# Download the exact public FLAN-T5-XL snapshot required for GRID embedding.
# Run under nohup; each safetensors shard resumes from its existing byte offset.
set -euo pipefail

target_dir="${1:-/data2/hnlcm/models/flan-t5-xl}"
revision="7d6315df2c2fb742f0f5b556879d730926ca9001"
base_url="https://hf-mirror.com/google/flan-t5-xl/resolve/${revision}"

mkdir -p "$target_dir"

download_small() {
  local file="$1"
  curl -fL --retry 20 --retry-delay 5 --connect-timeout 30 \
    -o "$target_dir/${file}.part" "$base_url/$file"
  mv "$target_dir/${file}.part" "$target_dir/$file"
}

download_shard() {
  local file="$1"
  curl -fL --retry 20 --retry-delay 5 --connect-timeout 30 --continue-at - \
    -o "$target_dir/$file" "$base_url/$file"
}

for file in \
  config.json \
  generation_config.json \
  model.safetensors.index.json \
  spiece.model \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json
do
  download_small "$file"
done

download_shard model-00001-of-00002.safetensors
download_shard model-00002-of-00002.safetensors

expected_shard_1=9449619912
expected_shard_2=1949477672
actual_shard_1="$(stat -c '%s' "$target_dir/model-00001-of-00002.safetensors")"
actual_shard_2="$(stat -c '%s' "$target_dir/model-00002-of-00002.safetensors")"

test "$actual_shard_1" -eq "$expected_shard_1"
test "$actual_shard_2" -eq "$expected_shard_2"

echo "FLAN-T5-XL download verified"
echo "revision=$revision"
du -sh "$target_dir"
stat -c '%n %s bytes' "$target_dir"/model-*.safetensors
