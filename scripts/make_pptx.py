"""Generate the NVIDIA pre-meeting deck as a native .pptx — pure stdlib.

No third-party deps (python-pptx is not installed, and the cluster blocks
login-node installs / offline compute nodes). A .pptx is just a ZIP of OOXML
parts, so we emit them directly with zipfile + string templates. The result is
an editable PowerPoint/Keynote/Google-Slides file.

Embeds the speedup bar charts we generated, dark theme, EN + KO bilingual.
Run via srun (CPU only); writes benchmarks/reports/deck/miniworld_nvidia_deck.pptx.
"""

import struct
import zipfile
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "reports" / "deck" / "miniworld_nvidia_deck.pptx"

INK = "0E1320"; INK2 = "161D2E"; WHITE = "FFFFFF"; DIM = "9AA6BE"
OURS = "E8412B"; NV = "76B900"; TEAL = "11A6A0"; GREY = "9AA3B2"; AMBER = "E8A23D"

EMU_IN = 914400
CX, CY = 12192000, 6858000  # 16:9
MX = 685800
FULL_W = CX - 2 * MX

IMG = {
    "trimul": ROOT / "src/miniworld_kernels/modules/triangle_multiplication/benchmark/trimul_forward_speedup.png",
    "transition": ROOT / "src/miniworld_kernels/kernels/transition/benchmark/transition_forward_speedup.png",
    "lnl": ROOT / "src/miniworld_kernels/kernels/layernorm_linear/benchmark/layernorm_linear_fwd_speedup.png",
}


def png_size(path: Path):
    data = path.read_bytes()
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _runs(runs):
    out = []
    for text, color, bold, sz in runs:
        b = ' b="1"' if bold else ""
        out.append(
            f'<a:r><a:rPr lang="en-US" sz="{sz}"{b} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="+mn-lt"/></a:rPr><a:t>{escape(text)}</a:t></a:r>'
        )
    return "".join(out)


def para(runs, align="l", space_after=600, bullet=False):
    bu = '' if bullet else '<a:buNone/>'
    return (
        f'<a:p><a:pPr algn="{align}" marL="0" indent="0">{bu}'
        f'<a:spcAft><a:spcPts val="{space_after}"/></a:spcAft></a:pPr>{_runs(runs)}</a:p>'
    )


def textbox(sp_id, name, x, y, cx, cy, paras, anchor="t"):
    body = "".join(paras)
    return f"""<p:sp><p:nvSpPr><p:cNvPr id="{sp_id}" name="{name}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
<p:txBody><a:bodyPr wrap="square" anchor="{anchor}"><a:normAutofit/></a:bodyPr><a:lstStyle/>{body}</p:txBody></p:sp>"""


def picture(sp_id, rid, x, y, cx, cy):
    return f"""<p:pic><p:nvPicPr><p:cNvPr id="{sp_id}" name="chart"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>"""


def slide_xml(shapes, has_img):
    r_ns = ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"' if has_img else ""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"{r_ns} xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{INK}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/><a:chOff x="0" y="0"/><a:chExt cx="{CX}" cy="{CY}"/></a:xfrm></p:grpSpPr>
{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def eyebrow(text):
    return para([(text, OURS, True, 1100)], space_after=300)


def title(text):
    return para([(text, WHITE, True, 3200)], space_after=300)


def rrect(idn, x, y, cx, cy, fill, text, tc="FFFFFF", sz=1000):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{idn}" name="b{idn}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 14000"/></a:avLst></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr anchor="ctr" wrap="square"/><a:lstStyle/>'
            f'<a:p><a:pPr algn="ctr"><a:buNone/></a:pPr><a:r><a:rPr lang="en-US" sz="{sz}" b="1">'
            f'<a:solidFill><a:srgbClr val="{tc}"/></a:solidFill><a:latin typeface="+mn-lt"/></a:rPr>'
            f'<a:t>{escape(text)}</a:t></a:r></a:p></p:txBody></p:sp>')


def row(start_id, y, h, boxes, gap=110000, sz=1000):
    """boxes: list of (label, fill, tc, weight). Lays them across FULL_W."""
    wsum = sum(b[3] for b in boxes)
    avail = FULL_W - gap * (len(boxes) - 1)
    x, idn, out = MX, start_id, []
    for label, fill, tc, wt in boxes:
        cx = int(avail * wt / wsum)
        out.append(rrect(idn, x, y, cx, h, fill, label, tc, sz))
        x += cx + gap; idn += 1
    return "".join(out), idn


def arch_slide(eyebrow_t, title_t, ko_t, base_boxes, ours_boxes, cap_en, cap_ko):
    sh = textbox(2, "t", MX, 520000, FULL_W, 1500000,
                 [eyebrow(eyebrow_t), title(title_t), para([(ko_t, DIM, False, 1300)], space_after=200)])
    nid = 3
    sh += textbox(nid, "bl", MX, 2120000, FULL_W, 300000, [para([("BASELINE — separate launches · 분리된 다중 launch", DIM, True, 1050)], space_after=0)]); nid += 1
    r, nid = row(nid, 2420000, 700000, base_boxes, sz=900); sh += r
    sh += textbox(nid, "ol", MX, 3450000, FULL_W, 300000, [para([("OURS — fused · 융합", OURS, True, 1050)], space_after=0)]); nid += 1
    r, nid = row(nid, 3750000, 820000, ours_boxes, sz=1000); sh += r
    sh += textbox(nid, "cap", MX, 4850000, FULL_W, 1600000,
                  [para([(cap_en, WHITE, False, 1300)], space_after=300), para([(cap_ko, DIM, False, 1200)])]); nid += 1
    return sh


def build_slides():
    """Each entry -> (list of shapes xml, image_key or None)."""
    slides = []
    full_w = FULL_W

    # 1 title
    sh = textbox(2, "t", MX, 1500000, full_w, 3200000, [
        eyebrow("H100 · bf16 · PRE-MEETING"),
        title("Kernel optimization with NVIDIA"),
        para([("three working tracks", NV, True, 3200)], space_after=500),
        para([("Today's pre-meeting fixes the agenda for the 7/2 offline session. "
               "Each track is backed by kernels already built and benchmarked.", DIM, False, 1500)], space_after=300),
        para([("오늘 사전 미팅에서 7/2 오프라인 세션의 의제를 정합니다. 각 트랙은 이미 만들고 벤치한 커널이 뒷받침합니다.", DIM, False, 1300)], space_after=600),
        para([("We build v1  →  ", OURS, True, 1700), ("NVIDIA optimizes v2", NV, True, 1700),
              ("    ·    1차 개발은 우리가 → 2차 최적화는 NVIDIA가", DIM, False, 1300)]),
    ])
    slides.append((sh, None))

    # 2 agenda
    body = [
        eyebrow("THE AGENDA · 의제"), title("What we want to agree on today"),
        para([("오늘 합의하려는 것", DIM, False, 1500)], space_after=500),
        para([("TRACK 01  ", OURS, True, 1600), ("Co-optimize the kernels — we build v1, NVIDIA optimizes v2 (fwd / bwd / inference, op by op)", WHITE, False, 1600)], bullet=False),
        para([("커널 공동 최적화 — 우리가 1차, NVIDIA가 2차 (forward / backward / inference, op 단위)", DIM, False, 1300)], space_after=500),
        para([("TRACK 02  ", OURS, True, 1600), ("AF3-like model inference — large inference on a single ordinary GPU (chunking, CPU offload, fused kernels)", WHITE, False, 1600)]),
        para([("AF3-like 모델 인퍼런스 — 일반 GPU 한 장에서 대규모 인퍼런스 (chunking, CPU offload, 융합 커널)", DIM, False, 1300)], space_after=500),
        para([("TRACK 03  ", OURS, True, 1600), ("Education & ramp-up — from AI-assisted to expert CuTeDSL (material, pairing, learn from each re-optimization)", WHITE, False, 1600)]),
        para([("교육 & 역량 강화 — AI 보조에서 CuTeDSL 전문 수준으로 (자료·페어링·재최적화 학습)", DIM, False, 1300)]),
    ]
    slides.append((textbox(2, "t", MX, 700000, full_w, 5800000, body), None))

    # 3 track1 scoreboard
    body = [
        eyebrow("TRACK 01 · CO-OPTIMIZE THE KERNELS"),
        title("Four kernels — vs the same default & vs NVIDIA's own"),
        para([("All forward, H100, bf16. Baseline = torch.compile'd PyTorch. · 전부 forward, 기준선 = torch.compile PyTorch.", DIM, False, 1300)], space_after=500),
        para([("Triangle Multiplication   ", WHITE, True, 1600), ("11.3× vs default", OURS, True, 1600), ("   · beats cuEquivariance & dt-v1 by 1.2–1.4× at every L", DIM, False, 1400)]),
        para([("Transition (SwiGLU)        ", WHITE, True, 1600), ("5.5× vs default", OURS, True, 1600), ("   · one fused kernel, h never touches HBM", DIM, False, 1400)]),
        para([("LayerNorm + Linear         ", WHITE, True, 1600), ("3.1× vs default", OURS, True, 1600), ("   · beats TransformerEngine at every (M, d)", DIM, False, 1400)]),
        para([("LayerNorm                  ", WHITE, True, 1600), ("6.0× vs default", OURS, True, 1600), ("   · cuEquivariance gives only ~1.0× at large d", DIM, False, 1400)]),
    ]
    slides.append((textbox(2, "t", MX, 700000, full_w, 5800000, body), None))

    # 4-6 kernel slides with images (left img, right bullets)
    def kernel_slide(eyebrow_t, title_t, ko_t, pts, img_key):
        img_x, img_y = MX, 1750000
        img_w = 5750000
        w, h = png_size(IMG[img_key])
        img_h = int(img_w * h / w)
        txt_x = MX + img_w + 400000
        txt_w = CX - txt_x - MX
        shapes = textbox(2, "t", MX, 600000, full_w, 1100000, [eyebrow(eyebrow_t), title(title_t),
                         para([(ko_t, DIM, False, 1300)], space_after=200)])
        shapes += picture(3, "rId2", img_x, img_y, img_w, img_h)
        body = []
        for tag, en, ko in pts:
            body.append(para([(tag + "  ", OURS, True, 1500), (en, WHITE, False, 1500)], space_after=200))
            body.append(para([(ko, DIM, False, 1250)], space_after=500))
        shapes += textbox(4, "b", txt_x, img_y, txt_w, 4600000, body)
        return shapes

    slides.append((arch_slide(
        "TRACK 01 · TRIANGLE MULTIPLICATION", "Architecture — two fused kernels around one bmm",
        "아키텍처 — bmm을 사이에 둔 두 개의 융합 커널",
        [("LN_in", GREY, WHITE, 1), ("left", GREY, WHITE, 1), ("right", GREY, WHITE, 1), ("gate", GREY, WHITE, 1),
         ("transpose", AMBER, WHITE, 1.3), ("bmm", TEAL, WHITE, 1), ("LN_out", GREY, WHITE, 1), ("out·gate", GREY, WHITE, 1)],
        [("KERNEL 1 · input proj (tm1)", OURS, WHITE, 2.3), ("cuBLAS bmm · L³", TEAL, WHITE, 1), ("KERNEL 2 · output proj (tm2)", OURS, WHITE, 2.3)],
        "Fusion: left+right become ONE gated GEMM; the bdll direct write removes the transpose (TMA); LN_in is folded into the GEMM.",
        "퓨전: 좌+우를 하나의 gated GEMM으로; bdll 직접 기록으로 transpose 제거(TMA); LN_in을 GEMM에 folding."), None))
    slides.append((kernel_slide(
        "TRACK 01 · TRIANGLE MULTIPLICATION", "Two fused kernels around one bmm",
        "bmm을 사이에 둔 두 개의 융합 커널",
        [("K1", "Input projection (tm1): LN_in + left/right in one gated GEMM, writing bdll directly (no transpose).",
          "입력 사영(tm1): LN_in + 좌/우를 하나의 gated GEMM으로, bdll 직접 기록 (transpose 제거)."),
         ("K2", "Output projection (tm2): LN_out + out-proj ⊙ gate, fused in the epilogue.",
          "출력 사영(tm2): LN_out + out-proj ⊙ gate를 epilogue에서 융합."),
         ("→", "Best at every L; 1.87 ms @ L=1024 vs cuEquiv 2.57.",
          "모든 L에서 최고; L=1024서 1.87 ms vs cuEquiv 2.57.")],
        "trimul"), "trimul"))

    slides.append((arch_slide(
        "TRACK 01 · TRANSITION (SwiGLU MLP)", "Architecture — fusion removes HBM round-trips",
        "아키텍처 — 융합으로 HBM 왕복 제거",
        [("LayerNorm", GREY, WHITE, 1), ("expand ↑4×", GREY, WHITE, 1), ("SwiGLU", GREY, WHITE, 1), ("squeeze ↓", GREY, WHITE, 1)],
        [("LN → expand ↑4× → SwiGLU → squeeze    (h stays in SMEM / registers)", OURS, WHITE, 1)],
        "Fusion: back-to-back fuse the whole MLP into ONE launch — the 4×-wide hidden h stays in SMEM, zero HBM round-trips (baseline does 3).",
        "퓨전: MLP 전체를 한 번의 launch로 b2b 융합 — 4× 넓은 hidden h가 SMEM에 상주, HBM 왕복 0회(baseline은 3회)."), None))
    slides.append((kernel_slide(
        "TRACK 01 · TRANSITION (SwiGLU MLP)", "One fused kernel — h never touches HBM",
        "한 개의 융합 커널 — h가 HBM에 가지 않음",
        [("1", "Naive path writes the 4×-wide hidden h to HBM between every stage (3 round-trips).",
          "naive 경로는 4× 넓은 hidden h를 단계마다 HBM에 기록 (3회 왕복)."),
         ("2", "Ours fuses LN → expand → SwiGLU → squeeze into one launch; h stays in SMEM.",
          "ours는 LN → expand → SwiGLU → squeeze를 한 번의 launch로; h는 SMEM 상주."),
         ("→", "~5.5× vs PyTorch; 1.24 ms vs 6.67 ms @ 1024.",
          "PyTorch 대비 ~5.5×; 1024서 1.24 ms vs 6.67 ms.")],
        "transition"), "transition"))

    slides.append((arch_slide(
        "TRACK 01 · LAYERNORM + LINEAR", "Architecture — LN folded into the GEMM epilogue",
        "아키텍처 — LN을 GEMM epilogue에 folding",
        [("LayerNorm (mean·rstd·normalize)", GREY, WHITE, 1.4), ("X̂ ↔ HBM", AMBER, WHITE, 1), ("Linear / GEMM  X̂·W", GREY, WHITE, 1.4)],
        [("GEMM tile loop:  mean/rstd → normalize+affine (epilogue) → X̂·W    (X̂ stays in registers)", OURS, WHITE, 1)],
        "Fusion: compute LN stats inside the GEMM tile loop and fold normalize+affine into the epilogue — X̂ never reaches HBM.",
        "퓨전: GEMM 타일 루프 안에서 LN 통계를 계산하고 normalize+affine을 epilogue에 folding — X̂가 HBM에 도달하지 않음."), None))
    slides.append((kernel_slide(
        "TRACK 01 · LAYERNORM + LINEAR", "LN folded into the GEMM epilogue",
        "LN을 GEMM epilogue에 folding — TransformerEngine 능가",
        [("1", "Baseline writes normalized X̂ to HBM; a GEMM reads it back.",
          "baseline은 정규화된 X̂를 HBM에 기록 → GEMM이 다시 읽음."),
         ("2", "Ours computes LN stats in-kernel, folds normalize+affine into the epilogue — X̂ never reaches HBM.",
          "ours는 LN 통계를 커널 내에서 계산해 epilogue에 folding — X̂가 HBM에 안 감."),
         ("→", "TE drops below the compiled-PyTorch baseline at large d; ours leads everywhere.",
          "큰 d에서 TE는 기준선 아래; ours는 전 구간 선두.")],
        "lnl"), "lnl"))

    # 7 track2 limits
    body = [
        eyebrow("TRACK 02 · AF3-LIKE MODEL INFERENCE"),
        title("The wall — and why existing fixes fall short"),
        para([("Single-GPU wall at ~1–3k residues (pair rep is O(N²); ≈100× more kernel launches than an LLM). · 단일 GPU 벽 ~1–3k residue.", DIM, False, 1300)], space_after=500),
        para([("Fold-CP", WHITE, True, 1500), (" — needs a multi-GPU cluster (dozens–256 GPUs); no help for single-GPU users · 클러스터 필요", DIM, False, 1400)]),
        para([("FastFold / AutoChunk", WHITE, True, 1500), (" — chunking is generic / op-agnostic (recompute overhead); mostly AlphaFold2 · 범용·op무관, AF2 중심", DIM, False, 1400)]),
        para([("LightNobel", WHITE, True, 1500), (" — a precision tradeoff from quantization · 양자화 정밀도 손실", DIM, False, 1400)]),
        para([("MegaFold / ScaleFold", WHITE, True, 1500), (" — training systems, limited inference benefit · 학습 시스템", DIM, False, 1400)]),
    ]
    slides.append((textbox(2, "t", MX, 700000, full_w, 5800000, body), None))

    # 8 track2 gap
    body = [
        eyebrow("TRACK 02 · THE GAP · 빈 자리"),
        title("Why go further — co-design, not generic patches"),
        para([("Each existing line solves one axis generically. None co-designs ", DIM, False, 1500),
              ("fused kernels + op-aware chunking for AF3 inference on a single GPU", OURS, True, 1500),
              (" — where the gains compound.", DIM, False, 1500)], space_after=500),
        para([("기존 방법들은 각자 한 축을 범용적으로만 풉니다. 융합 커널 + op별 chunking을 AF3 단일 GPU 인퍼런스에 공동설계한 사례는 없습니다.", DIM, False, 1300)], space_after=500),
        para([("1  ", OURS, True, 1500), ("Op-aware chunking / streaming of the O(N²) pair representation.  ·  O(N²) pair 표현의 op-인지 chunking.", WHITE, False, 1500)]),
        para([("2  ", OURS, True, 1500), ("CPU offload for cold tensors, overlapped with compute.  ·  cold 텐서 CPU offload, 연산과 오버랩.", WHITE, False, 1500)]),
        para([("3  ", OURS, True, 1500), ("Inference-only fused paths — fewer intermediates → less memory AND less recompute than chunking alone.", WHITE, False, 1500)]),
        para([("    인퍼런스 전용 융합 경로 — chunking만 할 때보다 메모리·recompute 둘 다 절감. cuEquivariance는 속도 한계가 아님.", DIM, False, 1250)]),
    ]
    slides.append((textbox(2, "t", MX, 700000, full_w, 5800000, body), None))

    # 9 track3
    body = [
        eyebrow("TRACK 03 · EDUCATION & RAMP-UP"),
        title("From AI-assisted to expert"),
        para([("Where we stand:  ", WHITE, True, 1500), ("Triton — write & autotune directly; CuTeDSL — used it, but AI-driven; we validate vs fp64 and read roofline.", DIM, False, 1450)]),
        para([("현재 수준: Triton 직접 작성·오토튠 / CuTeDSL은 AI 주도 / fp64 검증·roofline 해석 가능.", DIM, False, 1250)], space_after=500),
        para([("Where AI plateaus:  ", WHITE, True, 1500), ("cute ≈ Triton-level or simple quack variants; not yet fluent in warp specialization, TMA/WGMMA pipelining, epilogue fusion, occupancy budgeting.", DIM, False, 1450)]),
        para([("AI의 한계: cute가 Triton 수준 / warp specialization·TMA·WGMMA·epilogue 융합·occupancy는 미숙.", DIM, False, 1250)], space_after=500),
        para([("What would help:  ", WHITE, True, 1500), ("internal CuTeDSL / SM90 material + canonical patterns; light pairing / office hours; profile → next-lever guidance.", DIM, False, 1450)]),
        para([("요청: 내부 CuTeDSL/SM90 자료 + canonical 패턴, 가벼운 페어링, 프로파일 해석 가이드.", DIM, False, 1250)], space_after=500),
        para([("Learn from the diff — we build v1, NVIDIA re-optimizes to v2, we study what changed and why.  ·  diff로 배우기.", OURS, True, 1450)]),
    ]
    slides.append((textbox(2, "t", MX, 600000, full_w, 5900000, body), None))

    # 10 next
    body = [
        eyebrow("OUTCOME · 결과물"), title("From today to 7/2"),
        para([("Today  ", OURS, True, 1600), ("Agree the three tracks — scope, and which kernel / op leads each.  ·  세 트랙 합의.", WHITE, False, 1500)], space_after=400),
        para([("Before 7/2  ", OURS, True, 1600), ("Prepare evidence & questions — TriMul backward; a CuTeDSL question list.  ·  근거·질문 준비.", WHITE, False, 1500)], space_after=400),
        para([("7/2 offline  ", OURS, True, 1600), ("Work the diffs together — pick one kernel, do v1→v2 live.  ·  함께 diff 작업.", WHITE, False, 1500)], space_after=400),
    ]
    slides.append((textbox(2, "t", MX, 900000, full_w, 5200000, body), None))

    # 11 method
    body = [
        eyebrow("METHODOLOGY & SOURCES · 방법론·출처"), title("Honest caveats"),
        para([("· All baselines are torch.compile'd PyTorch (never eager).  ·  모든 기준선은 torch.compile PyTorch.", DIM, False, 1400)]),
        para([("· H100 80GB, bf16, B=1. TriMul forward-only; each kernel at its best regime. cuEquiv / dt-v1 reproduce team-gm's harness.", DIM, False, 1400)]),
        para([("· TriMul backward in development (within ~10% of dt-v1 at large L).  ·  TriMul backward 개발 중.", DIM, False, 1400)], space_after=500),
        para([("Sources: Fold-CP arXiv:2603.14806 · MegaFold 2506.20686 · FastFold/AutoChunk PPoPP'24 / 2401.10652 · LightNobel 2505.05893 · AF3 docs", DIM, False, 1150)]),
    ]
    slides.append((textbox(2, "t", MX, 900000, full_w, 5200000, body), None))

    return slides


# ---- OOXML scaffolding ----
RELNS = "http://schemas.openxmlformats.org/package/2006/relationships"

THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="mw">
<a:themeElements><a:clrScheme name="mw">
<a:dk1><a:srgbClr val="0E1320"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="161D2E"/></a:dk2><a:lt2><a:srgbClr val="E7ECF5"/></a:lt2>
<a:accent1><a:srgbClr val="E8412B"/></a:accent1><a:accent2><a:srgbClr val="76B900"/></a:accent2>
<a:accent3><a:srgbClr val="11A6A0"/></a:accent3><a:accent4><a:srgbClr val="F29A38"/></a:accent4>
<a:accent5><a:srgbClr val="9AA3B2"/></a:accent5><a:accent6><a:srgbClr val="E8A23D"/></a:accent6>
<a:hlink><a:srgbClr val="E8412B"/></a:hlink><a:folHlink><a:srgbClr val="F29A38"/></a:folHlink></a:clrScheme>
<a:fontScheme name="mw"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="mw">
<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
<a:lnStyleLst><a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
</a:fmtScheme></a:themeElements></a:theme>"""

BLANK_TREE = """<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree>"""

MASTER = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{INK}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>{BLANK_TREE}</p:cSld>
<p:clrMap bg1="dk1" tx1="lt1" bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>"""

LAYOUT = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank">{BLANK_TREE}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"""


def main():
    slides = build_slides()
    n = len(slides)

    sld_ids = "".join(f'<p:sldId id="{256+i}" r:id="rId{i+2}"/>' for i in range(n))
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
<p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{CX}" cy="{CY}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>"""

    pres_rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(n):
        pres_rels.append(f'<Relationship Id="rId{i+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>')
    pres_rels.append(f'<Relationship Id="rId{n+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    pres_rels_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">{"".join(pres_rels)}</Relationships>'

    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Default Extension="png" ContentType="image/png"/>',
          '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
          '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
          '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
          '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>']
    for i in range(n):
        ct.append(f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    ct.append('</Types>')

    root_rels = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
    master_rels = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
    layout_rels = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml)
        z.writestr("ppt/theme/theme1.xml", THEME)
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
        z.writestr("ppt/_rels/slideMaster1.xml.rels".replace("ppt/", "ppt/slideMasters/"), master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        used_imgs = {}
        for i, (shapes, img_key) in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(shapes, img_key is not None))
            if img_key:
                media = f"image_{img_key}.png"
                if img_key not in used_imgs:
                    z.writestr(f"ppt/media/{media}", IMG[img_key].read_bytes())
                    used_imgs[img_key] = media
                rels = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">'
                        f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                        f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media}"/></Relationships>')
            else:
                rels = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{RELNS}">'
                        f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {n} slides)")


if __name__ == "__main__":
    main()
