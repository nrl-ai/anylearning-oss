#!/bin/bash

# Check if source image is provided
if [ -z "$1" ]; then
    echo "Usage: ./make_icns.sh source.png"
    exit 1
fi

# Create iconset directory
mkdir icon.iconset

# Generate different sizes
sips -z 16 16     "$1" --out icon.iconset/icon_16x16.png
sips -z 32 32     "$1" --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     "$1" --out icon.iconset/icon_32x32.png
sips -z 64 64     "$1" --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   "$1" --out icon.iconset/icon_128x128.png
sips -z 256 256   "$1" --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   "$1" --out icon.iconset/icon_256x256.png
sips -z 512 512   "$1" --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   "$1" --out icon.iconset/icon_512x512.png
sips -z 1024 1024 "$1" --out icon.iconset/icon_512x512@2x.png

# Convert to icns
iconutil -c icns icon.iconset

# Clean up
rm -rf icon.iconset
