#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uchyEi039CTfPI_-sZ1HWGwOsLqGt2AT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uchyEi039CTfPI_-sZ1HWGwOsLqGt2AT" -O 'scripts/yelp_features.tar.xz' && rm -rf /tmp/cookies.txt
tar -xf scripts/yelp_features.tar.xz -C ./
rm -rf scripts/yelp_features.tar.xz
