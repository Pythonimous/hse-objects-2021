#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RTy3jS4HN2mH00G-4kLf5BWFtgl9ww0A' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RTy3jS4HN2mH00G-4kLf5BWFtgl9ww0A" -O 'scripts/yelp_photos.tar' && rm -rf /tmp/cookies.txt
tar -xf scripts/yelp_photos.tar -C ./data
rm -rf scripts/yelp_photos.tar
