# A100 SWA Atom DiT: Anthropic fullgraph compile

2026-10-05. 이전 [전체 모듈 비교](a100-module-comparison-20261005.md)의 SWA Anthropic 수치는 compile 실패 후 graph-only로 측정했다. 아래는 연결부를 수정한 뒤 세 backend를 모두 다시 측정한 결과다. 이전 기록은 그대로 보존한다.

Anthropic의 동적 import·row 선택·커널 launch를 primitive 단위 custom op 네 개로 등록하고 fake tensor 메타데이터를 제공했다. SWA 블록 전체를 opaque 처리하지 않았다. 선형층, modulation 주변 연산, RoPE, window index 생성 및 정렬은 Inductor graph에 남아 있다. `fullgraph=True`로 실제 Inductor executable이 실행됐으며, 18개 결과 모두 `compile_scope=module_forward:fullgraph`, `compiled=True`, CUDA graph manual이다. graph break를 허용하거나 eager fallback을 compiled로 표기하지 않았다.

원본 upstream revision은 `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`다. 런타임의 `dtk_kernels.py` SHA-256은 `92f2dc4285b6c7a4701ff1af6227fc5dc0ea6b3c5b26405bfb86c5cb2a594381`, `gather_attn.py`는 `4dbe4cd733b23d5afccc6d505b280be06d4b56e24223fb38befe832f92e5d60f`이며 import manifest와 일치한다. Gather K=129는 기존과 동일한 upstream candidate row이며 A100에서 로컬 검증한 composition이다. Anthropic이 제공한 별도 full-block 최적 구현이라는 뜻은 아니다.

## 조건과 결과

합의한 [shape protocol](../gpus/module-comparison-protocol.md)을 유지했다. A100 80GB PCIe / 300 W, B=1, A=1·5, atom N=1024·2048·4096, 폭 128, 4×32 heads, half-window 64, 한 블록, BF16 입력·가중치, TF32 허용, mask 0%, per-sample conditioning 및 block 내부 modulation이다. 공통 `benchmarks/runners/bench.py`의 warmup 이후 median ms다. 세 backend는 각 job에서 동일 GPU로 순차 실행했다. 각 shape의 노드·job·UUID·실제 tensor shape는 raw provenance에 보존했다. 속도비는 baseline / MiniWorld다.

| A | Atom N | MiniWorld compiled ms | Anthropic compiled ms | PyTorch compiled ms | Anthropic / MiniWorld | PyTorch / MiniWorld |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 0.0737 | 0.3983 | 0.1321 | 5.40× | 1.79× |
| 1 | 2048 | 0.0840 | 0.4567 | 0.2488 | 5.44× | 2.96× |
| 1 | 4096 | 0.0973 | 0.6789 | 0.5325 | 6.98× | 5.47× |
| 5 | 1024 | 0.1004 | 0.7158 | 0.2181 | 7.13× | 2.17× |
| 5 | 2048 | 0.1352 | 1.1069 | 0.6410 | 8.19× | 4.74× |
| 5 | 4096 | 0.2376 | 2.0306 | 1.9118 | 8.55× | 8.05× |

## 검증 및 재현

선택한 GPU 검증은 **9 passed, 80 deselected**다(Slurm job 63967, 230.69 s). 신규 fullgraph/거부 정책 3개와 기존 SWA attention·SWA DiT·SwiGLU eager/reference/graph 회귀 6개를 포함한다. 신규 연결부 및 테스트 파일의 Ruff 검사도 통과했다. 벤치마크 jobs는 63961–63964다.

`tests/integrations/test_anthropic_swa_compile_gpu.py`는 BF16/N128 및 FP16/N384에서 eager warmup 이전 fullgraph compile을 실행한다. 비영점 modulation/gate 가중치로 독립 PyTorch FP32 reference와 비교하며, residual을 제외한 update의 상대 오차도 확인한다. 입력·conditioning·QKV 및 modulation 가중치·RoPE·mask를 변경한 뒤 재컴파일 없이 CUDA graph를 재실행한다. 내부 mask gap과 완전 padding row가 유효 row로 바뀌는 경우를 포함한다. GPU graph 출력은 compiled 출력과 bitwise 일치해야 하며, compiled 대 eager 상대 오차 <1%, 독립 reference 대비 출력 <4%, update <6%를 요구한다. 실제 fullgraph 실행과 네 upstream primitive 및 linear 노드의 존재를 확인한다.

성능 harness는 기존 protocol과 동일하게 기본 초기화 가중치를 사용한다. 초기 zero gate로 블록 출력이 identity가 되므로 harness의 출력 검증만으로 정확성을 판정하지 않는다. 위 별도 비영점 가중치 검증으로 이를 보완했다. 학습 지원은 추가하지 않았으며 기존 inference-only 거부 정책을 유지한다. 원본 gather attention은 FP32 value를 지원하지 않으며, eager와 compiled 모두 동일한 `v-dtype-float32` 거부 사유가 전달되는지 별도로 검증했다.

```bash
python -m pytest -q tests/integrations/test_anthropic_swa_compile_gpu.py \
  tests/integrations/test_anthropic_modules_a100_gpu.py \
  -k 'fullgraph_ or (module_reference_and_graph and (swa or swiglu))'

# Slurm compute allocation 안에서 pinned upstream/Pixi 환경을 로드한 후 실행.
# A=1,5 각각에 대해 128..256과 512..512 두 번 실행한다. 실제 atom N=seq_len*8.
python benchmarks/runners/bench.py target=swa_dit level=module mode=inference \
  'implementations=[miniworld,pytorch,anthropic]' compile=true cudagraph=manual \
  min_seq_len=128 max_seq_len=256 seq_len_step=128 n_augment=1 \
  precision=bf16-mixed allow_tf32=true mask_prob=0.0 n_layers=1 ++shared_cond=false
```

Source hash: `444f8c341315984c7769d64600ca7e13dcac7d2b811b0a9ced9defef59957966`.

[Raw CSV](a100-swa-anthropic-compile-20261005.csv), [JSON 및 provenance](a100-swa-anthropic-compile-20261005.json).
