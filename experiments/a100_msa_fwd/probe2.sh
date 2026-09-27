#!/bin/bash
echo "=== base"; python bench.py --op pwa_split --length 384 768 2>&1 | grep -E "^pwa|^    |Error|error" | head -4
echo "=== ctr FRAGDB"; python bench.py --op pwa_split --length 384 768 -D PWA_C_FRAGDB=1 2>&1 | grep -E "^pwa|^    |Error|error" | head -4
echo "=== value JMAJOR"; python bench.py --op pwa_split --length 384 768 -D PWA_V_JMAJOR=1 2>&1 | grep -E "^pwa|^    |Error|error" | head -4
