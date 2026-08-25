# 커널 명명 규칙 (확정)

> 규칙 자체는 현행입니다. 규칙을 설명하려고 인용한 **옛 이름들은 기록**이며,
> 지금 존재하는 커널 목록은 `src/miniworld_engine/kernels/registry.csv`입니다.

    <func>_<role>[_<detail>]_<backend>

세 칸 모두 필수(detail만 선택). 소문자 snake_case. 아래 고정 어휘만 사용한다.

원칙: **이름은 그 커널이 계산하는 것을 말한다.** 파일이 어디 있는지, 어느 경로가 부르는지,
누가 먼저 썼는지는 이름이 아니다.

## 1. `<func>` — 맨 앞. 이 커널이 구현하는 연산 (11개)

    layernorm            LN 통계 리덕션을 커널 안에서 수행하는 LayerNorm 계열
    layernorm_linear     LN + Linear GEMM을 한 커널에서 수행
    adaln                Adaptive LayerNorm (cond -> scale/bias 생성 및 적용)
    transition           Transition MLP (expand-SwiGLU-squeeze)
    cond_transition      Conditioned Transition (AdaLN + Transition)
    trimul               TriangleMultiplication in-projection 스테이지
    trimul_outproj       TriangleMultiplication out 스테이지 (LN + 게이티드 출력 투영)
    triangle_attention   Triangle Attention
    augmented_attention  Augmented (atom) Attention
    bias_only_attention  Bias-only Attention
    gated_projection     y = sigma(gate) * proj 계열 (게이트 단독 / GEMM 융합 모두)

버린 이름과 이유:

    tm1              코드는 in-projection이고 outgoing이 아니다 -> trimul
    tm2              코드는 out 스테이지이고 incoming이 아니다  -> trimul_outproj
    fused_ln_mask    layernorm_fwd의 row_scale 플래그일 뿐 (본문 동일)
    fused_ln_gate    adaln + role epilogue 로 표현된다
    sigmoid_gate     gated_projection + role gate 로 표현된다

`trimul_in` / `trimul_out`은 금지한다 — outgoing/incoming으로 오독된다. outgoing과 incoming의
구분은 어떤 커널에도 없다: 두 스테이지 사이 `torch.bmm` 인자 전치에서만 일어난다.
`fused_` 접두는 정보가 없으므로 func에 쓰지 않는다.

## 2. `<role>` — 이 커널이 담당하는 계산 단계. 조각을 **실행 순서대로** 이어 쓴다

    fwd bwd            순/역전파 전체 (한 커널이 다 함)
    bwd_pre            역전파 전처리 (delta/rowsum/dscale 버퍼 생성)
    bwd_reduce         파셜 -> 최종 리듀스
    dx dw dbias dlnw   그래디언트 대상. dlnw = LN 의 gamma/beta
    dq dk dv dkdv      어텐션 그래디언트 조각
    stats              mean/rstd 등 통계만
    gemm               행렬곱 본체만
    gate               sigmoid 게이트 곱 (y = sigma(a) * b)
    sigmoid            GEMM 뒤 sigmoid 만 적용 (곱은 다른 커널)
    swiglu             SiLU(a) * b
    expand squeeze     Transition 의 확장/축소 GEMM
    layernorm          커널 안에서 수행하는 LN 단계
    epilogue           본 GEMM 밖의 후처리 단계
    transpose          레이아웃 변환만
    fold               가중치/텐서 프리폴드

두 규칙:

- **`gate`는 sigmoid 출력게이트로 고정한다. SwiGLU는 반드시 `swiglu`다.**
  transition 계열이 SwiGLU를 `gate`라고 불러서 cond_transition 의 출력게이트와 같은 단어를
  쓰고 있었다 — 같은 단계가 다른 이름을, 다른 단계가 같은 이름을 갖는 원인이었다.
- **순서가 곧 의미다.** `gemm_gate` = GEMM 후 게이트 곱, `gate_gemm` = 게이트 곱 후 GEMM.

`dlnw`가 `dw`와 따로 있는 이유: adaln 에는 가중치 계열이 둘(LN gamma, Linear W_scale/W_bias)이고
서로 다른 텐서를 계산하는 별개 커널이라, 둘 다 `dw`가 되면 "같은 알고리즘"이라는 거짓 주장이 된다.

## 3. `<detail>` — 알고리즘·계약 변종 (17개)

**같은 `<func>_<role>` 커널이 2개 이상 실존해 구분이 필요할 때만 붙인다.** 하나뿐이면 안 붙인다.
detail 은 "런치 인자로 표현되는 차이"가 아니라 **호환되지 않는 계약**을 가리킨다. 판정 기준은 실측이다:
같은 인자를 먹지 못하거나, 같은 인자에 다른 값을 내면 그 축은 실재한다.

알고리즘 / 구조

    recompute    저장 activation 대신 통계·출력에서 순전파 값을 재구성
    split        프로그램별 파셜 버퍼 + 별도 reduce 커널 (persistent 그리드 포함)
    atomic       최종 누산기에 tl.atomic_add
    b2b          back-to-back GEMM (중간 텐서 미저장)
    ktiled       가중치 K 방향 타일링
    packed       여러 오퍼랜드/출력을 한 버퍼에 인터리브·연접
    inplace      입력 버퍼에 되쓴다, 출력 버퍼가 없다 (autotune restore_value 필요)
    dropres      행 브로드캐스트 dropout scale / residual add 를 게이트 에필로그에 융합
    rowscale     LN 출력에 행별 스칼라(마스크)를 곱한다 — 필수 오퍼랜드
    noaffine     아핀(gamma/beta) 미적용 — 가중치 포인터가 파라미터에 없다
    foldstats    mean 대신 프리폴드된 c1 = mean*rstd 통계를 읽는 계약
    saveact      같은 수학 + backward 용 활성/통계 버퍼를 추가로 쓰는 **별도 커널**.
                 방향은 하나로 고정: **저장하는 쪽이 토큰을 받는다.** `nosave` 는 쓰지 않는다.
                 같은 커널이 constexpr 플래그(SAVE_GATE/SAVE_PREACT)로 처리하면 붙이지 않는다.

레이아웃 / 인덱싱 — **레이아웃 계약은 런치 인자가 아니라 의미의 일부다**

    mmajor       (D,M) 채널메이저 평면을 읽거나 tl.trans 로 그렇게 쓴다
    strided      텐서마다 독립적인 (row,col) 스트라이드 쌍. 무표시형은 스트라이드를 공유한다고 가정
    flat         1-D 선형 인덱스. 원소 수 하나만 받고 전 오퍼랜드 contiguous 가정.
                 무표시형은 행x열 타일 (M, N, stride, GROUP_M)
    contig       스트라이드 인자가 없거나 특정 축 stride=1 하드코딩 — strided view 를 주면 오답

정밀도 / 아키텍처

    fp32         계산을 fp32 로 승격 (GEMM 은 allow_tf32=False)
    sm90 sm100   아키텍처 전용 경로. **파일·디렉터리 이름의 sm100 은 근거가 아니다**
    extern       외부 벤더(quack/CUTLASS) 커널 — 우리 알고리즘과 같다는 주장을 하지 않기 위한 표시

detail 이 둘 이상이면 이 순서로 쓴다:

    알고리즘/구조 -> 레이아웃 -> 정밀도 -> 아키텍처 -> extern

`mmajor` 가 실재한다는 근거: `layernorm_transpose_dbn` 과 `layernorm_linear_cute_sm100_ln_mmajor`
는 AST 노드 수·줄 수가 완전히 같고 서로는 비트 동일인데, row-major LN forward 들과는 같은 입력에
`Y rel=1.442e+00` 로 다른 값을 낸다. 코드가 같아도 레이아웃 계약이 다르면 다른 커널이다.

버린 detail 과 이유:

    lowp         지칭 대상이 없다. 이 토큰을 만들게 한 커널은 fp32 업캐스트가 빠져 bf16 에서
                 컴파일조차 되지 않았다 -- 정밀도 변종이 아니라 버그였고, 업캐스트를 넣으니
                 기존 커널과 동일해져 병합됐다. 코드만 읽으면 변종처럼 보이고 띄우면 드러난다.
    nosave       saveact 와 같은 축의 반대 방향. 방향은 "저장하는 쪽에 표시"로 고정
    persistent   split 에 흡수. persistent_bwd 와 partial_bwd 는 파라미터 철자만 다른 같은 커널이었고
                 (dx/part_dw/part_db 비트 동일) 이미 병합됐다
    privatized   지칭 대상이 하나도 없다. 실물이 나오면 되살릴 것
    transpose    role 에 이미 있다. 레이아웃 축은 mmajor 가 담당
    memeff       실제 메커니즘은 atomic 또는 recompute
    lowreg       본문이 기본 커널과 동일했고 비트 동일로 병합 완료
    fused        전부 constexpr 플래그이거나 b2b
    stacked      packed 로 통합
    pairbias     호출부 이름. 실제 차이는 fp32 dot -> fp32

## 4. `<backend>`

    triton | cute | cuda | cutlass

맨 끝, 정확히 하나. **디렉터리가 아니라 실제 구현 언어를 따른다.** 현재 `cute/` 디렉터리 아래에
`@triton.jit` 커널이 9개 있다(`fused_ln_mask/cute/`, `transition/cute/`,
`layernorm_linear/cute/ln_linear_sm100.py` 등) — cute 경로를 보조하는 triton glue 이며 backend 는
`triton` 이다. 파일 이동을 권장한다.

## 5. 최우선 원칙 — 같은 알고리즘이면 같은 이름

`<func>_<role>[_<detail>]` 이 동일하고 `<backend>` 만 다르면 "이 둘은 같은 수학·같은 알고리즘을
서로 다른 언어로 구현한 것"이라는 **주장**이다.

따라서 **같은 backend 안에서 이름이 충돌하면 그건 복붙 중복이라는 증거다.** 새 detail 을 만들어
회피하지 말고 커널을 합쳐라.

**constexpr 플래그나 런치 인자로 표현되는 차이는 이름을 얻지 못한다.** 구체적으로 다음은 새 이름의
근거가 될 수 없다:

    활성(mean/rstd, x_hat, gate, preact) 저장 여부
    affine(gamma/beta) 적용 여부
    in/out 스트라이드 분리
    in-place 여부
    오퍼랜드 패킹 주소지정, contiguous 가정

근거는 리포 자신에 있다: `_gate_mul_kernel`(trimul/gate_elem.py)과 `_bidir_front_kernel`은 이미
`SAVE_GATE` / `SAVE_PREACT` constexpr 플래그로 저장 여부를 처리한다. 같은 일을 이름으로 처리한
커널들이 그 옆에 복붙본으로 남아 있었다.

## 6. 금지

- 파일명/변수명/출처를 이름에 쓰지 말 것.
  `main`, `launch`, `composed`, `front`, `back`, `train`, `infer`, `dtv1`, `quack`, `te`,
  `from_scratch` 는 이름이 아니다.
  특히 `front`/`back` 은 방향이 아니라 파이프라인 위치여서 fwd/bwd 와 충돌한다 —
  `trimul_back`(`_back_kernel`)은 **순전파** 커널인데 옆의 `trimul_back_fused_dconcat`은
  **역전파** 커널이다.
- `layer_norm` / `layernorm` 혼용 금지 -> 항상 `layernorm`.
- backend 토큰이 이름 중간에 오는 것 금지.
- 순번 이름(`ln1`, `ln2`, `fused3`) 금지.

## 7. 런처는 커널이 아니다

호스트측 디스패처/래퍼는 커널 이름공간 밖이다. 대신 **그 런처가 띄우는 모든 커널을 개별로
이름표에 올린다.** 예: `layer_norm_bwd_cuda`(파이썬 런처)는 이름 2개가 필요하다 —
`layernorm_bwd_split_cuda` + `layernorm_bwd_reduce_cuda`.

이름표는 파일 기준이 아니라 **launch 지점 기준**으로 열거한다. 파일 기준으로 세면 `.cu` 커널과
cute collective 클래스가 통째로 빠진다(현재 최소 6개 CUDA 커널이 이름표에 없다 —
[naming-audit.md](../docs/kernels/naming-audit.md) §2 참조).

---

## 커널 등록 규칙 (필수)

새 커널은 **`src/miniworld_engine/kernels/registry.csv`에 행을 추가해야 완료된다.** 이름만 규칙에
맞추는 것으로는 부족하다. 열은 9개다.

    kernel, backend, family, kind, level, file, symbol, driver, check

`kind`와 `level`은 **개발자가 직접 채운다.** 자동으로 채워지지 않는다.

| 열 | 값 | 의미 |
|---|---|---|
| `kind` | `gemm` / `reduce` / `elem` | 행렬곱을 하는가(`tl.dot`, cuBLAS gemm, MMA/WGMMA/tcgen05), 행렬곱 없이 레인·블록 간 감축을 하는가(`tl.sum/max/min`, warp shuffle, atomic 누적), 아니면 원소 단위 작업인가 |
| `level` | `atom` / `token` / `both` | 모델에서 원자 단위 블록에 쓰이는가, 토큰·페어 단위에 쓰이는가, 양쪽인가 |
| `driver` | `module:function` | 그 커널을 실제 입력으로 한 번 실행하는 함수. 비워둘 수 없다 |
| `check` | `module:function` | 같은 형상으로 다시 실행해 torch fp32 레퍼런스와 대조하는 함수. 어떤 카드에서도 launch되지 않는 커널만 비워둘 수 있다 |

### 왜 필수인가

**행이 없으면 커널은 "미검증"이 아니라 "존재하지 않는 것"이 된다.** `autotune.run_all`, 커버리지
보고, 모든 정확도 스윕이 registry를 순회한다. 예전에는 AST로 `configs_for(...)`를 훑어 커널을
찾았는데, 그건 답을 추론하는 방식이었고 양방향으로 틀렸다 — 그래서 선언이 생겼다.

**`level`은 기계가 채울 수 없다.** 그 커널이 모델의 어디에 쓰이는지는 커널 텍스트의 성질이 아니라
아키텍처의 성질이다. 나중에 자동으로 메울 수 없으므로 추가하는 사람이 넣어야 한다.

**`driver`/`check`가 비면 구멍이 보이지 않는다.** driver는 커널이 도는지를, checker는 수치가 맞는지를
증명한다. 이 둘은 다른 얘기다 — 커널 56개가 "예외를 내지 않았다"는 이유로 `ok`로 기록돼 있었고, 그
상태에서 마스킹 버그 3개가 프로덕션 코드에 남아 있었다. 그 셋은 우연히 checker가 있어서 잡혔다.

`kind`에서 `reduce`를 `elem`과 따로 두는 이유: 감축에는 **누적 순서**가 있다. 타일 크기가 수치를
바꾸는 자리, 배리어 누락이나 tail 마스크 오류가 행 전체를 오염시키는 자리가 전부 여기다. 실제로
발견된 결함 4개 중 3개가 감축·수축축에서 나왔다.

### `dtypes` — 그 커널이 어떤 dtype으로 튜닝되어야 하는가

`level`에 따라 결정된다.

| level | dtypes |
|---|---|
| `token` | `bf16` |
| `atom` | `bf16\|fp32` |
| `both` | `bf16\|fp32` |

`level`에서 유도하지 않고 **별도 열로 둔다**. `level`은 커널이 *어디에 쓰이는지*를, `dtypes`는
*무엇으로 튜닝되어야 하는지*를 말하는 서로 다른 사실이다. 유도해버리면 나중에 어떤 커널이 다른
집합을 필요로 할 때 그렇게 말할 방법이 없고, 매핑을 한 번 바꾸면 기존 모든 행이 조용히 다시
쓰인다.

캐시는 이미 shape bucket과 **별개로** dtype으로 분할된다(`autotune/capture.py`의 `_dtype_of`).
그래서 이 열은 "완결된 빌드가 무엇을 덮어야 하는가"의 **선언**이다 — fp32 엔트리가 없을 때 그것이
아무도 예상 못 한 부재가 아니라 **보이는 구멍**이 되게 하는 것이 이 열의 일이다.

### 강제

`tests/test_registry_complete.py`가 검사한다. GPU가 필요 없다.

* 빈 셀 (`check` 제외), 어휘 밖의 `kind`/`level`, 한 family 안에서 갈리는 `level`
* **`dtypes`가 `level`을 따르는지**, 그리고 `bf16`/`fp32`/`fp16`의 `|` 결합인지
* 중복 이름, 존재하지 않는 `file`
* **`configs_for("op")`를 호출하는데 행이 없는 op** — 잊은 커널을 실제로 잡는 방향이다. config는
  있고 행은 없는 커널은 프로덕션에서 돌면서 어떤 스윕에도 나타나지 않는다
* `driver`/`check`가 가리키는 함수가 그 모듈에 정의돼 있는지
* **손으로 넣은 `kind`가 소스와 맞는지** — `src/miniworld_engine/tools/classify.py`가 커널이 인라인하는
  호출의 전이 폐쇄를 따라가 판정한다. 불일치는 자동으로 행의 잘못이 아니다(`tl.dot`이 생겼으면
  커널의 kind가 실제로 바뀐 것이다). 그래서 테스트는 양쪽을 다 출력하고 판단은 사람에게 남긴다.
  손으로 선언하고 기계가 검증한다.

`src/miniworld_engine/tools/classify.py`는 본문만 보지 않는다. flash-attention forward는 `tl.dot`을
인라인되는 `@triton.jit` 헬퍼에 두고, CUDA LayerNorm은 감축을 `__device__` 헬퍼에 둔다 — 본문만
읽으면 둘 다 `elem`으로 잡힌다.
