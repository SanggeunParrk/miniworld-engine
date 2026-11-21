# FlashBias (thuml, NeurIPS 2025, MIT) - NOT APPLICABLE, not connected

FlashBias computes attention with a bias that is (or is compressed to) low rank: `softmax(q k^T + q_bias k_bias^T) v` with explicit factors `q_bias`, `k_bias`
(`flash_bias_func(q, k, v, q_bias, k_bias, ...)`). The pair bias of triangle attention / AttentionPairBias is `to_bias(LN(pair))`, a dense `[H, N, N]` logit plane of full rank
(a learned function of the pair representation); no exact factors exist. FlashBias in AlphaFold 3 (4_AlphaFold3/) obtains its 1.5x by an SVD / neural low-rank APPROXIMATION of that
bias, which changes the values (not exact, no engine-accuracy-standard claim). Recorded as not applicable; nothing installed on the cards (the pinned checkout is for reading only).
