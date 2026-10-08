#!/usr/bin/env bash
# Downloads the publicly available scenes used in the paper into scenes/downloaded/.
#   Sponza Atrium (Marko Dabrovic, 76K tris)  - McGuire Computer Graphics Archive
#   Conference room (282K tris)                - McGuire Computer Graphics Archive
#   Armadillo (345K tris)                      - Stanford 3D Scanning Repository
# Erw6, Soda Hall and the plant are not public; procedural stand-ins are used instead.
# Please respect each archive's license/terms of use.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=scenes/downloaded
mkdir -p "$DEST"

fetch() {  # url output
  if [ -s "$2" ]; then echo "have $2"; return; fi
  echo "downloading $1"
  if command -v curl >/dev/null; then curl -fL --retry 3 -o "$2.part" "$1"; else wget -O "$2.part" "$1"; fi
  mv "$2.part" "$2"
}

unzip_to() {  # zip dir
  mkdir -p "$2"
  if command -v unzip >/dev/null; then unzip -oq "$1" -d "$2"
  else python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; fi
}

fetch https://casual-effects.com/g3d/data10/research/model/dabrovic_sponza/sponza.zip "$DEST/dabrovic_sponza.zip"
unzip_to "$DEST/dabrovic_sponza.zip" "$DEST/sponza"

fetch https://casual-effects.com/g3d/data10/research/model/conference/conference.zip "$DEST/conference.zip"
unzip_to "$DEST/conference.zip" "$DEST/conference"

fetch http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz "$DEST/Armadillo.ply.gz"
if [ ! -s "$DEST/armadillo/Armadillo.ply" ]; then
  mkdir -p "$DEST/armadillo"
  gunzip -c "$DEST/Armadillo.ply.gz" > "$DEST/armadillo/Armadillo.ply"
fi

echo "scenes in $DEST:"
find "$DEST" -name '*.obj' -o -name '*.ply' | sort
