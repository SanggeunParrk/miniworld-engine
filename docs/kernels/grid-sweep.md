# 전체 그리드 스윕 (205,266 config) — 설계와 준비

`gen_grid.py`가 만드는 그리드를 **손으로 줄이지 않고 그대로** 튜닝하기 위한 기록.
왜 이 결정이 가능한지, 그리고 실제로 돌리기 전에 무엇이 깨져 있었는지를 남긴다.

## 1. 그리드

접두사로 역할을 정하고(`BLOCK_M*`→M, `BLOCK_N*`→N, `BLOCK_K*`→K, 그 외→E, `GROUP_M`→GROUP),
kind(gemm/reduce/elem)별 값 집합의 데카르트 곱 × `num_warps` × `num_stages`.

| kind | ops | 조합 | 비중 |
|---|---:|---:|---:|
| gemm | 48 | 173,016 | 84% |
| reduce | 19 | 20,880 | 10% |
| elem | 24 | 11,370 | 6% |
| **합** | **91** | **205,266** | |

크기 분포: 10k+ 4개 / 5k–10k 4개 / 2k–5k 20개 / 1k–2k 23개 / 500–1k 22개 / <500 18개.
**축이 4개인 gemm op 8개가 전체의 45%**(각 5,832–15,552, 합 91,368)를 차지한다.
`15,552 = 288(타일) × 54(warps 6 × stages 9)`.

`num_warps`는 2의 거듭제곱만 가능하다(Triton `AssertionError`; warp group은 정의상 2의 거듭제곱이고
`warpsPerCTA`가 차원별 2의 거듭제곱으로 인수분해된다). 32까지 유효하다(1024 스레드).
`num_stages`에는 **컴파일러 상한이 없다.** 비용이 stage당 operand tile 하나로 정확히 선형이므로
상한은 `smem_limit / operand_tile`이고 config마다 다르다 (sm86 101,376 B / sm80 167,936 B /
sm90·sm100 232,448 B). 그래서 stage는 넓게 두고 컴파일 실패로 걸러낸다.

## 2. 샤딩

`builder.py`는 이미 **shape**(`--case/--dims/--length/--mode`)로 샤딩하지만, 그 샤드들은 전부
**동일한 full grid**를 들고 있었다. 20만 개는 그렇게 못 나누므로 **config set 자체**를 쪼갠다
(생성기 `tools/gen_shards.py`는 그 뒤 삭제됐다 — config set을 쪼개 배포하는 방식을 쓰지 않게 됐고, 지금은 `configs/grid/`가 손으로 관리되는 사다리다. 아래 두 조건은 그 시절 검증 결과의 기록이다.)

샤드 디렉터리가 반드시 만족해야 하는 두 가지 (생성 결과 + **실제 GPU 실행**으로 검증함):

- **COMPLETE** — 대상 op뿐 아니라 **모든 op**의 CSV를 담아야 한다. `run_case`는 모듈 전체를
  돌리므로 이 샤드가 튜닝하지 않는 op도 launch된다. CSV가 없으면 config 리스트가 비고, Triton이
  자기 `Config({})`로 대체하며, 커널은
  `dynamic_func() missing required positional arguments`로 죽는다.
  → 비대상 op에는 filler 1행. (`shard-0000` 실측: 88 op registered, **missing 0**)
- **IN-GRID** — filler는 `autotune/configs/accuracy`가 아니라 **그 op 자신의 생성 그리드**에서 가져온다.
  `merge_shards`는 샤드들이 들고 있던 grid의 **합집합**을 해싱하므로, 그리드 밖 filler가 섞이면
  합집합이 실제 그리드를 넘어서고 해시가 full-grid 실행과 달라진다 →
  `store_ranked_configs`가 해시 불일치에 **entries 전체 리셋**으로 답한다.
  (실측: 91/91 op이 샤드에서 full grid를 정확히 복원, 그리드 밖 filler **0**)

분할 기준은 op 개수가 아니라 **조합 수**다. 8개 op이 45%를 차지하므로 op당 1샤드는 대부분의 잡을
놀리면서 몇 개만 며칠씩 돌게 만든다. `--per-shard 8000` → 26 샤드 (min 5,266 / max 8,000).

### 하드웨어 end-to-end 검증

일회성 `shard_e2e` 프로브(이후 삭제) — 실제 모듈 실행으로 세 가지를 한 번에 확인했다.
disjoint한 두 config 샤드를 각각 별도 프로세스로 돌리고(config dir 선택이 import 시점이므로
같은 프로세스에서는 불가능) 병합한다. A6000 결과:

```
case gated_projection fired 1 op(s): ['gated_projection_gate_triton']   <- 이름을 고르지 않고 발견
--- shard 0: grid=6 buckets=['bfloat16|R=128,shape_key=256']
--- shard 1: grid=6 buckets=['bfloat16|R=128,shape_key=256']
merge: union of shard grids = 12 configs -> hash 77a1a01b710b
       cache config_space_hash            = 77a1a01b710b   HASH MATCHES UNION: True
       degenerate 'any' buckets: none
```

1. filler 덕에 두 샤드 모두 launch 성공 2. bucket이 `any|any`가 아닌 실제 값
3. 병합 해시 = **두 샤드 grid의 합집합** 해시 (고치기 전이면 한쪽 샤드의 6-config 해시)

op을 이름으로 고르려다 두 번 틀렸다(`layernorm_linear_pair_bias`는 아무것도 안 쏘고,
`gated_projection`은 `gate_gemm`이 아니라 `gate_triton`을 쏜다). case의 dispatch는
dims/length/mode에 달려 있으므로 **한 번 돌려보고 발견**하게 바꿨다.

## 3. 돌리기 전에 고쳐야 했던 것들

`fcd3c7a`("unify kernel names…")가 `make_cache_prune` / `early_config_prune` 배선을 걷어내면서
그것에 딸려 있던 것들이 **조용히** 같이 죽었다. 전부 실패 없이 통과하던 것들이라 발견이 늦었다.

| 증상 | 실제 결과 |
|---|---|
| `_record_one`의 (dtype, bucket)이 prune 객체에서 왔음 | prune이 사라진 뒤 전부 `any|any`. 저장소의 모든 캐시가 op당 엔트리 **1개** |
| kernel 38개 파일이 `key_bucket_of`/`tensor_dtype_of`를 import만 하고 호출 안 함 | `kernels/**/triton/*.py`가 ruff 제외라 lint가 못 봄 |
| `build/audit.py`가 같은 죽은 속성을 읽음 | op 이름 0/83, bucket key 0/83 해석. 전 op에 "not introspectable" 경고만 내고 검증한 게 없음 |
| `tests/builder/test_build_matrix.py`가 삭제된 `_is_compile_monster`를 import | 모듈 자체가 collect 불가 → **빌드 매트릭스 규칙 테스트 20개도 같이 부재** |
| `tests/numerics/test_int64_offsets.py`가 리터럴 `tl.arange(0, BLOCK_M)`을 고정 | 같은 커밋의 `BLOCK_M`→`BLOCK_M1` 개명에 실패. 지키려던 `.to(tl.int64)`는 멀쩡했음 |

`any|any`가 왜 위험한가: `shape_key`는 모든 커널의 `key=[...]`에 들어 있으므로 **Triton은
in-process에서 shape bucket별로 재튜닝한다.** 느려지지 않으니 아무도 눈치채지 못한다. 잃어버리는
것은 **디스크로 나가는 캐시**뿐이다 — op당 config 하나가 모든 shape을 덮고, 마지막에 벤치된 shape이
이긴다. 이 상태로 20만 개를 돌렸다면 shape key 작업(token 128–512, atom 256–8192, dim 64–768)의
결과가 통째로 뭉개진 채 나왔을 것이다.

이제 bucket은 커널이 선언한 `autotuner.keys`에서 직접 유도한다. per-kernel 배선이 없으므로 83개
autotuner 전부에 자동 적용되고, **Triton이 실제로 재튜닝하는 기준과 어긋날 수 없다** —
손으로 쓴 쪽은 어긋날 수 있었고, 실제로 어긋났다.

### merge_shards의 grid 합집합

`merge_shards`는 `grid`를 **가장 먼저 읽힌 샤드**에서 가져왔다. 지금까지의 샤드는 전부 shape 분할이라
grid가 동일했으므로 우연히 맞았고 아무도 잡아내지 못했다. config set을 쪼개면 샤드마다 grid가 다르고,
병합된 캐시는 어떤 full-grid 실행도 재현하지 못하는 해시를 기록한다 → 다음 빌드가 결과를 통째로
버린다. 읽은 순서에 따라 해시가 달라지기까지 했다. 지금은 signature로 dedup한 **합집합**을 해싱한다.

## 4. 실측: bucket 수와 실제 작업량

autotuner는 `key=[...]`의 **값 조합마다** config 리스트 전체를 다시 벤치한다. 따라서 실제 작업량은
`sum_op grid(op) × buckets(op)`이고, `buckets(op)`는 추측하면 안 된다.

- 커널 key 리스트의 데카르트 곱 → **25.6M**. 심한 과대추정. N/K/ND/DC는 같이 움직이고
  (ND = 4D, DC는 모델당 고정), boolean key 상당수는 매트릭스에서 한 값만 쓴다.
- 커밋된 캐시를 세면 → op당 **1개**. 이건 측정이 아니라 위 버그의 잔해였다 (전부 `any|any`).

그래서 직접 쟀다 (일회성 프로브로 측정 후 삭제, A6000, 빌드 매트릭스 **2085 unit 전수**,
op당 config 1개 — 어느 bucket에 떨어지는지는 어떤 config를 벤치하든 같으므로):

```
ran 1625, failed 460, 435s        (skip: trimul 324 + trimul_bidir 126 + 10)
ops captured 47
buckets/op  min 1  median 3  max 21  mean 3.8  total 177
```

bucket은 `shape_key` 단독이 아니라 **다른 key 차원과의 조합**이다:
`H2=1024,K=512,shape_key=384`, `ADD_RESIDUAL=1,N=128,USE_DROPOUT=0,shape_key=256` 처럼.
그래서 승수가 실재한다. 최다는 `trimul_gemm_gate_mmajor_triton` **21개**.

주의: 47/91 op만 잡혔다. trimul 계열 460 unit이 skip되었기 때문이고, 이 값들은 **하한**이다.

## 5. 실측: 컴파일 생존율과 config당 비용

`.bench/gridcost.py` — `adaln_gemm_gate_triton`(15,552, 최대 op)의 그리드에서 **균등 샘플 400개**
(앞에서 N개를 자르면 전부 가장 작은 타일 = 가장 싸고 가장 잘 살아남는 쪽만 본다). A6000 sm86.

```
survival 60.5%   (거부 158건 전부 OutOfResources(smem))
compile  p50=0.57s  p75=1.82s  p90=11.96s  p95=57.6s  p99=392.6s  max 454.6s
bench    0.029s/config
60초 넘는 config가 전체 컴파일 시간의 72%를 먹는다
```

**분포가 전부다.** median 0.57초인데 mean은 10초다. 비용의 거의 전부가 컴파일이고(벤치는
config당 0.03초로 무시 가능), 그 컴파일 비용의 **72%가 60초 이상 걸리는 소수의 config**에 있다.
즉 `capture.py`의 60초 fork+SIGKILL 예산은 안전장치가 아니라 **가장 큰 비용 레버**다.

| | serial | 16 컴파일 워커 | × 4.4 bucket |
|---|---:|---:|---:|
| 예산 없음 | 575 GPU-h | 35.9 GPU-h | 158 GPU-h |
| **60초 예산** (현행) | 316 GPU-h | **19.8 GPU-h** | **87 GPU-h** |
| 30초 예산 | 209 GPU-h | 13.0 GPU-h | 57 GPU-h |

30초로 줄이면 34% 더 아끼지만 **p95가 57.6초**다 — 5%의 config가 그 위에 있고 그 중에 우승자가
있으면 잃는다. 60초는 두 분포(정상 수 초 vs 스필 수 분)를 깨끗이 가르므로 그대로 둔다.

거부가 전부 smem인 것은 좋은 소식이다: 자원 할당 단계에서 실패하므로 ptxas까지 가지 않는다.

**결론: 60초 예산 + 16 워커 + 26 샤드로 shape bucket 하나당 ~20 GPU-h, 실측 평균 4.4개
bucket 기준 ~87 GPU-h.** GPU 8장이면 벽시계 ~11시간이다.

## 6. 남은 것

- trimul 계열 skip 원인 → 48/91 op만 측정된 이유이자, bucket 수가 하한인 이유
  (525건 중 130건은 `d_hidden != d_pair`로 triton 구현이 지원 안 하는 조합 — 버그 아님)
- b2b 커널의 독립적인 K축 2–3개를 따로 튜닝할지 (최대 27× 감축 여지)
