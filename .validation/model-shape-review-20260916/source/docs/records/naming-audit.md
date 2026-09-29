# 커널 이름 정리 감사 — 발견된 결함

> **기록입니다, 현행 문서가 아닙니다.** 111개 커널을 `docs/kernels/naming.md` 규칙으로
> 재명명하던 시점의 감사 결과입니다. 여기 나오는 옛 이름들이 이 문서의 *주제*이므로
> 그대로 둡니다. 현재 이름은 `src/miniworld_engine/kernels/registry.csv`가,
> 옛 이름 -> 새 이름 대응은 `docs/kernels/rename-map.tsv`가 정본입니다.

명명 규칙(`.bench/NAMING.md`)에 맞춰 111개 커널의 이름을 다시 짜는 과정에서, 이름이 아니라
**코드가** 틀린 것들이 드러났다. 이름을 "실제 계산하는 것"으로 맞추려면 코드를 읽어야 하고,
읽으면 이런 게 나온다. 아래는 이름 문제가 아닌 실제 결함만 모은 것이다.

## 1. triangle_attention/atomic.py — 결함 2개, 둘 다 수정·검증 완료

이 파일의 backward는 **한 번도 실행된 적이 없었다**. `bench_kernel_tri_attn_bwd`가 존재하지 않아
triangle_attention의 bwd 커널 4개(`_bwd_dq`, `_bwd_dkdv`, `_bwd_preprocess`, `_atomic_bwd`)는
정확도 스윕에도 컴파일 검증에도 걸리지 않는다. 그래서 결함이 겹쳐 쌓여 있었다.

**(a) 컴파일 불가.** `_attn_bwd`의 파라미터를 `H` → `HL`로 개명할 때 시그니처만 바꾸고 호출부를
안 고쳤다. 440행이 `_attn_bwd_dqdkdv(..., H, ...)`로 스코프에 없는 `H`를 넘겨 triton JIT에서
NameError. 받는 쪽 `H` 파라미터는 본문에서 한 번도 읽히지 않는 죽은 인자여서 둘을 함께 삭제했다.

**(b) 범위 밖 읽기.** (a)를 고치니 그 뒤에 숨어 있던 결함이 드러났다 — 그래디언트 4개가 전부
NaN. `compute-sanitizer memcheck`가 지점을 특정했다:

    Invalid __global__ read of size 16 bytes
      at atomic.py:249:_attn_bwd_preprocess
      Address ... is out of bounds

이 파일의 커널들은 **스트라이드 인자를 받지 않는다**. `_attn_bwd_preprocess`는 주소를
`off_hz*HEAD_DIM*N_CTX + off_m*HEAD_DIM + off_n`으로 계산한다. forward는 이 계약을 스스로
지킨다(`q, k, v, bias = [x.contiguous() for x in ...]`). 그런데 **backward는 `grad_output`에
그걸 하지 않았다.** `out.sum().backward()`는 autograd가 stride-0 expanded 텐서를 넘기므로 실제
저장 공간이 1원소인데 커널은 `L*D`개를 연속으로 읽는다. `main.py`가 면역인 이유는 그쪽
preprocess가 명시적 5D 스트라이드를 받기 때문이다(주석에 "no .contiguous()"라고 적혀 있다).

`main.py`와 대조하며 진단하는 과정에서 잘못 지목한 원인이 둘 있었다. 기록해 둔다:
- **`dk`/`dv`/`dbias`의 `empty_like`**: 아니다. 버퍼를 NaN으로 채우고 돌려도 전부 덮여 쓰인다.
- **`M`을 bf16으로 왕복시키는 것**: 아니다. fp32로 고친 코드도 여전히 NaN이었다. 다만 이건
  별개의 정당한 개선이라 남겼다 — `M`은 log2 공간 logsumexp이고 backward가 이걸로
  `p = exp2(qk*scale - M)`를 재계산하므로 bf16의 유효자리 3자리가 지수에 실린다. `main.py`는
  fp32로 유지한다. **NaN과는 무관하다.**

증상이 헷갈렸던 이유: NaN 개수가 실행마다 달라져서(dk 16384/24576/28672) 경합처럼 보였다.
실제로는 범위 밖 읽기가 인접 메모리 내용을 읽은 것이라 할당자 상태에 따라 결과가 달라진 것이다.

**검증** (`.bench/smoke_fixes.py`, 벤치를 거치지 않고 커널을 직접 띄운다):

    compute-sanitizer   수정 전 Invalid read ×14  →  수정 후 0 errors
    main.py vs atomic.py   dq rel=1.05e-2  dk=6.5e-3  dv=0  dbias=6.5e-3

bf16 eps가 약 7.8e-3이므로 이 편차는 반올림 범위다. 즉 두 파일이 **같은 역전파를 다르게
분해한 것**이라는 명명 전제가 실측으로 확인됐다 — 고치기 전에는 검증할 수 없던 주장이다.

## 1b. layernorm fwd가 가중치를 x의 컬럼 스트라이드로 인덱싱 (수정 완료)

`kernels/layernorm/triton/main.py`
`layer_norm_fwd_fused`가 길이 N의 1-D 벡터인 `W`/`B`를 `tl.load(W + cols * stride_c)`로 읽었다.
`stride_c`는 **x의 컬럼 스트라이드**이므로 가중치와 아무 관계가 없다. 같은 파일의 bwd는
`W + cols`로 올바르게 읽고 `stride_wc`/`stride_bc`를 별도 인자로 받기까지 한다 — 저자도
가중치 스트라이드가 별개임을 알고 있었다. `lowreg.py`도 올바르다.

모든 런처가 `.contiguous()`를 거쳐 `stride_c == 1`이라 드러나지 않았다. 4개 로드를
`W + cols`/`B + cols`로 고쳤고, 현재 호출자에게는 비트 동일하다.

부수: 같은 파일 bwd의 `stride_wc`, `stride_bc`는 본문에서 읽히지 않는 죽은 인자다(호출부 3곳이
계속 넘기고 있다).

## 2. 실행 경로 없는 커널 72개

58-task 배열로 `.ops` 레코드를 수집해 브랜치→커널 귀속을 실측했다 (`.bench/attrib.tsv`).
43개 벤치 브랜치 중 32개가 커널을 launch, **커널 111개 중 39개만 귀속 확인**.
나머지 72개는 벤치로 도달할 방법이 없다 — 정확도도 성능도 검증된 적이 없다는 뜻이다.

계측기 한계 하나: launch 레코더는 Triton autotuner만 후킹하므로 cute/cuda 커널은
원리적으로 안 보인다. `layernorm/quack_cute` 브랜치가 에러 없이 "ops 0개"로 나온 이유다.

**그리고 111이라는 숫자 자체가 하한선이다.** `devices.py`는 `configs_for()` op(=triton)과 bench가
import하는 심볼만 열거하므로, cute collective 클래스와 `.cu` 커널은 원리적으로 안 잡힌다.
확인된 누락:

| 파일 | 이름표에 없는 `__global__` 커널 |
|---|---|
| `layernorm/cuda/layer_norm_cuda_kernel.cu` | `layer_norm_fwd_kernel`, `layer_norm_bwd_main_kernel`, `layer_norm_bwd_reduce_kernel` |
| `transition/cuda/transition_cuda_kernel.cu` | `cast_kernel`, `swish_mul_kernel`, `transition_grad_kernel` |

이름표에는 이 6개 대신 파이썬 런처 `layer_norm_bwd_cuda` **하나**만 올라와 있었다. cute 쪽도
같은 이유로 `GemmLnGatedSm90`, `GemmDLnGatedSm90`, `SwiGLUExpandKernel`, `SwiGLUGateBwdKernel`,
`_TransitionDabLNBwdSm90`, `_DgradLNBwdSm90` 등이 빠져 있고, `adaln/cutlass/*.cu`와
`conditioned_transition/cutlass/*.cu`도 마찬가지다.

열거 기준을 파일이 아니라 **launch 지점**으로 바꿔야 실제 커널 집합이 나온다.

## 2b. 매니페스트가 triton 커널 9개를 cute로 표기 (수정 완료)

`autotune/devices.py`가 backend를 **디렉터리 경로**로 판정했다(`/cute/` in path → cute).
그런데 `configs_for(...)`로 찾은 op은 정의상 전부 `@triton.autotune`이다. `cute/` 디렉터리에
사는 triton 커널 9개(`fused_ln_mask`, `layernorm_linear_cute_sm100_ln_mmajor`,
`tm1_cute_gate_mul`, `tm1_cute_glu_wide`, `transition_cute_cdup_interleave`,
`transition_cute_xn_recompute`, `transition_sm100_grad_mul`,
`trimul_cute_front_sm100_transpose`, `trimul_sm100_glu_bdll`)가 전부 cute로 잘못 표기됐다.
`cute/`에 있다는 건 "cute 경로를 보조하는 triton glue"라는 뜻이지 backend가 아니다.

고친 뒤: 102 triton + 8 cute + 1 cuda = 111. triton 102개는 세트당 config CSV 102개와 일치한다.

같은 파일에서 함께 고친 것: bench 행의 파일 경로에 패키지 이름이 빠져 `fused_ln_mask`가
중복 계수되던 것(112→111), 패키지 import가 `__init__.py`로 안 풀리던 것, 사라져 있던
`observed_requirements`, (target, impl) 시절 잔재 `classify`/`_family_files`/`_module_file`.

## 3. 아키텍처 요구사항으로 위장한 결함

`aug_attn_compute_efficient` 브랜치는 `ModuleNotFoundError`로 죽는다 —
`miniworld_engine.kernels.augment...` 모듈이 존재하지 않는다. 아키텍처 문제가 아니다.
(`observed_requirements`의 기호 목록에 `compute_`가 들어 있어서 이게 "SM 요구사항"으로
오분류되고 있었다. 기호를 `arch=compute_`로 좁혀 고쳤다.)

## 4. 중복은 39개가 아니라 5쌍이었다 — 측정으로 확정

처음에 "커널 102개 중 39개가 복붙본"이라고 적었다. **틀렸다.** 그 숫자는 구역 에이전트들의
"같은 수학을 계산한다"는 판단을 내가 "같은 코드다"로 옮겨 적은 것이었다. 세 단계로 검증했다.

**(1) 유사도 측정은 실패했다.** 네 번 고쳐 만들었고 매번 사각지대가 있었다 — 첫 등장 순서
alpha-rename은 파라미터 하나가 끼면 뒤 이름이 전부 밀리고(저장 1줄만 다른 쌍이 0.000),
`ast.walk`은 폭 우선이라 순서가 섞이고, 순서 기반은 교환법칙 재배열을 놓치고, 노드 다중집합은
"같은 원시연산으로 만들어졌다"까지만 말한다.

**(2) 정적 증명은 1쌍만 확정했다.** 교환법칙 피연산자 정렬 + 본문 순회 기준 이름 정규화 후
AST 완전 일치를 요구하면 긍정은 증명되지만 부정은 증명되지 않는다. `w` vs `w[None, :]`,
임시변수 인라인 같은 의미 보존 재작성과 진짜 차이를 정적으로 가를 수 없다.

**(3) 런타임 등가 프로브가 답을 줬다** (`.bench/equiv_probe.py`). 커널 A가 실제로 launch될 때
인자를 가로채, 같은 입력을 형제 커널에 replay하고 변한 버퍼를 비교한다. 인자의 역할을 추론하지
않는다(실행 전후로 변한 것이 곧 출력). replay는 위치가 아니라 **파라미터 이름**으로 맞춘다 —
후보 쌍들은 인자 개수가 다르므로 위치 기반은 조용히 어긋난다.

여기서도 제 버그를 두 번 고쳤다. 오토튜너가 config에서 주입하는 `BLOCK_*`를 "호출자가 안 준
인자"로 세서 판정 64건을 전부 기각했고, 그다음엔 **비교가 0건인데 EQUIVALENT로 찍는** 결함이
있었다(`worst`가 0으로 초기화된 채 갱신 없음). 지금은 비교가 없으면 `INCONCLUSIVE`를 낸다.

### 병합한 5쌍 (전부 소스에서 제거, GPU 검증 완료)

| 제거된 커널 | 대체 | 근거 |
|---|---|---|
| `layernorm_lowreg_fwd` | `layer_norm_fwd_fused` (HAS_ROWSCALE=False) | Y·Mean·Rstd 3/3 BITWISE |
| `augmented_attention_memeff_fwd` | main.py `_attn_fwd` | M·Out 2/2 BITWISE, 양방향 |
| `augmented_attention_memeff_bwd_preprocess` | main.py `_attn_bwd_preprocess` | 정적으로 동일 프로그램 |
| `layernorm_linear_cute_sm100_ln_mmajor` | `_ln_transpose_dbn_kernel` | Y 1/1 BITWISE |
| `adaln_fused3_gemm_gate_train` | `_gemm_gate_kernel` + `SAVE_GATE` | Y 1/1 BITWISE |

triton op 102 → 97, 커널 111 → 106. config CSV 25개, `axes.csv` 12행, 캐시 1개 정리.

병합 중 실제 위험 두 개를 처리했다. `augmented_attention`의 두 fwd는 bias 스트라이드
파라미터 순서가 `bz,bh,bm,bn` vs `bz,bm,bh,bn`으로 뒤바뀌어 있었고, 호출부가 `*bias.stride()`로
위치 splat을 하고 있어서 import만 바꾸면 조용히 어긋난다 — 스트라이드를 명시적으로 재배치했다.
`SAVE_GATE`는 오토튠 키에 넣었다(같은 파일의 `HAS_W` 관행).

### 코드가 같은데 커널은 다른 사례

    layernorm_main_fwd → layernorm_transpose_dbn                Y: rel=1.442
    layernorm_main_fwd → layernorm_linear_cute_sm100_ln_mmajor  Y: rel=1.442
    fused_ln_mask      → layernorm_transpose_dbn                out_ptr: rel=1.442

`layernorm_transpose_dbn`은 `sm100_ln_mmajor`와 AST 노드 수·줄 수가 완전히 같고(797/797,
37/37) 서로는 비트 동일인데, row-major LN forward들과는 rel=1.442로 다르다. m-major 입력을
읽는 계약이기 때문이다. **레이아웃 계약은 런치 인자가 아니라 의미의 일부다** — 조정 단계에서
`mmajor`/`strided`/`contig`를 "런치 인자로 표현되니 이름을 얻지 못한다"며 버린 판정은 이 숫자로
반박된다.

### 판정하지 못한 것

96쌍은 `NOT_COMPARABLE`이다 — 같은 인자를 먹지 않는다. `triangle_attention_fwd`와
`atomic_fwd`는 파라미터가 하나도 겹치지 않고, `adaln_infer_ln1`은 `Cond`/`CondAff`/`LnW`를
요구한다. 이건 "확인 실패"가 아니라 **서로 대체 불가**라는 결과다.

그리고 `gated_projection_gate` 후보 7개는 **어떤 커널도 벤치 실행 경로가 없어** 프로브가
발동조차 못 했다. 같은 elementwise 게이트로 보이는 구현이 5개 모듈에 흩어져 있는데 하나도
검증된 적이 없다 — 이름 문제가 아니라 커버리지 문제다.

## 5. detail 토큰이 파일 출처를 베껴서 동작과 어긋난 것

규칙 §6이 금지한 것("파일명을 그대로 쓰지 말 것")을 어긴 사례. 이 4개에는 원자연산도,
메모리 절약 기법도 없다 — 그냥 `atomic.py`/`memory_efficient.py`에 들어 있을 뿐이다:

- `triangle_attention_atomic_fwd`
- `triangle_attention_atomic_bwd_preprocess`
- `augmented_attention_memeff_fwd`
- `augmented_attention_memeff_bwd_preprocess`

그리고 `memory_efficient`는 **재계산이 아니다**. 절약 대상은 backward의 중간 버퍼다:
기본 경로의 `dq_expand`(num_splits배 fp32) + 큰 `dbias` 대신 `dq`/`dbias` 하나에 atomic 누산한다.
forward 재계산량은 기본 경로와 동일하다. 즉 메커니즘은 `atomic`이며, 그렇게 이름 붙이면
triangle의 atomic bwd와 detail이 자동으로 일치한다 — 이게 이번 정리의 요점이다.

`bias_only_fused_gemm`의 `fused`는 정반대의 오칭이다. softmax를 torch가 미리 계산해 두고
GEMM만 도는, flash forward보다 **덜** 융합된 커널이다.

## 6. 삭제하면 안 되는 "죽은" 코드 — 의도적 신호등

처음에 삭제 후보로 분류했으나, 파일을 읽어보니 **저자가 실패한 접근의 기록으로 일부러 남긴
것**이었다. 지우면 같은 실수를 다시 하게 된다.

`kernels/bias_only_attention/triton/fused.py` (`bias_only_fused_gemm`)
docstring 첫 줄이 `"NEGATIVE RESULT, kept as a signpost"`이고, "strided-gather로 permute를
피한다는 아이디어는 진다. Do NOT revive"까지 근거와 함께 적혀 있다.

`kernels/trimul_inproj/triton/back_fused.py` (`trimul_back_fused_dconcat5`)
유일한 호출자가 `"NEGATIVE RESULT — tried & not adopted (kept as reference)"`라고 적어 두었다.

**진짜 문제는 다른 것이다**: 신호등인데도 5개 config 세트에서 autotune 대상으로 남아
빌드 시간을 계속 먹는다. `bias_only_fused_gemm`이 45분 컴파일의 정체다
(`acc = tl.zeros([BLOCK_M1_J, BLOCK_M1_I * BLOCK_K])` — N extent가 튜닝 축 두 개의 **곱**이라
all-64 조합이 64×4096 fp32가 된다). 코드는 남기고 **튜닝 대상에서 빼는 것**이 맞다.

같은 표시가 붙은 세 번째 항목 `transition_b2b_ktiled`는 docstring이 "UNVERIFIED"라고 적었지만
실제로는 `K > 128 && save_xn=False` 경로에서 프로덕션으로 쓰인다 — 주석이 낡은 것이다.

## 7. 이름 외 부수 정리 대상

- `adaln` 4개 파일 전부 `key_bucket_of`, `tensor_dtype_of`를 import하고 쓰지 않는다.
- `adaln/triton/main.py:15` `AUTOTUNE = settings.current().autotunes("adaln")` — 사용처 없음.
- `adaln_bwd_input_kernel`의 `USE_BF16`/`USE_FP16` constexpr은 본문에서 읽히지 않는다
  (이 커널은 `input_precision="ieee"` 고정). 호출부는 계속 넘겨서 불필요한 컴파일 분기만 만든다.
- `augmented_attention/main.py` `_attn_bwd`의 `reset_to_zero=['DQ','DBias']`는 잔재다.
  본문이 둘 다 `tl.store`로 전량 덮어쓰므로, autotune 시행마다 큰 `dq_expand`를 0으로
  채우는 비용만 낸다.
- `atomic.py` forward는 backward용 저장 시 logsumexp 통계를 `M.to(torch.bfloat16)`로 내렸다가
  되돌린다 (main.py는 fp32 유지). 두 경로의 수치가 갈리는 지점이다.
- 버킷 인자 이름이 `seq_group`(adaln/fused3.py)과 `GROUP_M`(나머지)으로 갈려 있다.
  `fused3.py`에서 `GROUP_M`은 GEMM의 L2-swizzle 튜닝 축이라 의미가 겹친다 — `seq_group`으로
  통일해야 한다. 관련: [l2-swizzle.md](l2-swizzle.md)

## 8. 명명 스펙 자체의 사실 오류 (수정 필요)

`.bench/NAMING.md`의 `<func>` 고정 목록에 이렇게 적었다:

    tm1                  TriangleMultiplication outgoing
    tm2                  TriangleMultiplication incoming

**코드와 어긋난다.** `kernels/tm1/*`는 전부 **in-projection**(σ(x@Wg)·(x@Wp), 좌/우)이고
`kernels/tm2/*`는 전부 **out-stage 게이티드 투영**(σ(x_gate@Wg)·(x_out@Wo))이다. outgoing과
incoming의 구분은 커널이 아니라 그 사이의 `torch.bmm` 인자 전치
(`trimul_inproj/triton/bidirectional.py:177-178`)에서만 일어난다.

따라서 `tm1 → trimul`(in-projection), `tm2 → trimul_outproj`. `trimul_out`/`trimul_in`은
outgoing/incoming으로 오독되므로 쓰면 안 된다.

같은 종류의 오류: `front`/`back`도 방향이 아니라 파이프라인 위치(bmm 앞단/뒷단)를 뜻한다.
그래서 `trimul_back`(`_back_kernel`)은 **순전파** 커널인데 바로 옆 `trimul_back_fused_dconcat`은
**역전파** 커널이다. 같은 `back_` 접두가 fwd와 bwd를 동시에 덮고 있다.

## 9. 어휘가 중복 코드를 떠받치고 있다

이번 작업의 구조적 발견. 규칙 §5("같은 알고리즘이면 같은 이름, backend만 다름")를 적용하면
**같은 backend 안에서 이름이 충돌하는 순간이 곧 복붙 중복의 증거**가 된다. 그래서 5개 구역이
독립적으로 요청한 새 어휘 15개 중 상당수는 알고리즘 축이 아니라 **중복본을 서로 구분하려고
필요해진 것**이었다:

| 어휘 | 존재 이유 |
|---|---|
| `contig` | `triangle_attention_fwd`와 `atomic_fwd`가 주소계산만 다른 같은 커널 |
| `lowreg` | `layernorm_main_fwd`와 `lowreg_fwd`가 문장 단위로 동일 |
| `flat` | 같은 elementwise를 1-D/2-D 인덱싱으로 두 번 구현 |
| `strided` | 같은 LN fwd를 스트라이드 인자만 늘려 다시 구현 |

어텐션 구역이 직접 그렇게 적었다: "triangle atomic.py의 중복을 제거하면 `contig`는 불필요하다."

**결론: 중복 제거가 개명보다 먼저다.** 중복을 남긴 채 이름을 붙이면 그 중복을 정당화하는
어휘를 스펙에 영구히 새기게 된다.
