#!/usr/bin/env python
"""
Generate docs/gazer_report.html — the GAZER step 1-5 progress report.

Data sources (all project files, regenerated on every run):
  logs/diagnostics/gazer_bench.json         stress test (step 2)
  logs/diagnostics/gazer_eval_compare.json  same-metric comparison (step 3a, optional)
  logs/gazer_grpo_backbone_t5.log           VAL + GRPO stat history (step 3)
  git tag/log                               version management (step 1)

Usage:  nnunet_env/bin/python tools/gazer_report.py
Re-run after gazer_eval_compare.py finishes to fold in the comparison numbers.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "gazer_report.html")
BENCH_JSON = os.path.join(ROOT, "logs", "diagnostics", "gazer_bench.json")
EVAL_JSON = os.path.join(ROOT, "logs", "diagnostics", "gazer_eval_compare.json")
TRAIN_LOG = os.path.join(ROOT, "logs", "gazer_grpo_backbone_t5.log")

# ---------------------------------------------------------------------------
# Code excerpts shown in the report (verbatim from the project sources).
# Kept as plain strings (no f-string) so braces stay literal.
# ---------------------------------------------------------------------------
CODE_BENCH = """# tools/gazer_bench.py — 검증 방법: 모듈 격리 타이밍 + FLOPs + 스트리밍 FPS
def time_fn(fn, iters=30, warmup=10, dtype=torch.bfloat16):
    # CUDA 이벤트 타이밍의 중앙값(ms) (bf16 autocast, warmup 10 + iters 30)
    ...

# B. 모듈별 wall time — 인코더 특징을 미리 계산해 decoder만 분리 재측정
with torch.no_grad():
    e1 = model.enc1(xf) ... b = model.bottleneck(p3)   # 인코더 1회만
def run_decoder():
    outs = []
    for t in range(T):
        d3 = model.up3(bm[:, t])
        d3 = model.dec3(torch.cat([d3, e3a[:, t]], dim=1))
        d2 = model.up2(d3)   ...  d1 = model.dec1(...)
        outs.append(model.final_cls(d1))
    return torch.stack(outs, dim=0)
t_dec = time_fn(run_decoder)      # -> "cnn_decoder_56to448_T5"

# C3. 스트리밍 경로 (시연 120Hz path, B=1, T=1 decode-last, FIFO cache)
def stream_one():
    model.reset_cache()
    for _ in range(20):
        outs.append(model(x_stream, output_all_frames=False))
    return outs
t_stream = time_fn(stream_one, iters=5, warmup=2) / 20.0
# FPS = 1000 / t_stream ; mamba OFF 아블레이션도 동일 경로로 재측정
"""

CODE_GRPO = """# rl/grpo.py — 신호 강화: per-pixel dense advantage / logp
def dense_pixel_advantage(rewards_pix, adv_clip=5.0):
    # A = clip(r - mean_g r, +-adv_clip)  (std 정규화 없음)
    # 확신 픽셀: G개 샘플 전부가 동일 클래스 -> A = 0 정확 (플러 0)
    # 경계 픽셀만 O(1) -> grad이 teacher 불일치 위치에만 작용

def categorical_logp_dense(logp, m):     # [B,C,H,W] logp, [G,B,H,W] m
    lp = logp.unsqueeze(0).expand(m.shape[0], *logp.shape)
    return lp.gather(2, m.unsqueeze(-1)).squeeze(-1)   # [G,B,H,W]

# losses/grpo_loss.py — GRPOLoss.dense (아키텍처 불변, 경로만 dense 전환)
ratio = torch.exp((logp - logp_ref).clamp(-20, 20))
pg   = -(ratio * adv).mean()             # per-pixel policy gradient
loss = pg + beta_kl * kl.mean()          # per-pixel exact KL 앵커 (β: 0.05→0.01)
"""

CODE_EVAL = """# tools/gazer_eval_compare.py — 동일 지표 원샷 비교 (val-only landscape)
# Sequence 분할은 GT 프레임이 0개 -> reward는 SAM3 landscape에서만 오므로
# val 프레임(16,800개, ~27GB RAM, ~2h)만 precompute하면 충분 (전체 4.4배 비용 절감)
cache = precompute_reward_landscape(cfg, args.config,
        loaders["validation"].dataset, None, logf)

for tag, ckpt in [("gazer_ep9",    GAZER_BEST),          # GRPO 8ep 정책
                  ("init_fold1",   cfg.model.pretrained_checkpoint),  # ep0
                  ("baseline_400", BASELINE_400)]:       # 수도라벨 시퀀스 학습
    score = eval_val(model, val_loader, cache, cfg, dev, tag)
# eval_val == train_gazer.validate 재현: bf16 forward + dice_agreement(
#   softmax(logits), SAM3 soft4) — 지표(val/agreement_teacher)를 1:1으로 고정
"""

CODE_DISTILL = """# models/vivim_backbone_multiframe.py — 경량화(knob): 인코더+디코더 채널 축소
self.enc_base = base_channels if enc_base is None else int(enc_base)
eb = self.enc_base                            # enc1/2/3 = eb*{1,2,4}
self.enc1 = ConvBlock(in_channels, eb); self.enc2 = ConvBlock(eb, eb*2)
self.enc3 = ConvBlock(eb*2, eb*4)
self.bottleneck = ConvBlock(eb*4, base_channels * 8)
#   출력 폭 base*8 고정 = Mamba3 temporal 입력(학습된 temporal capacity 상속)
#   enc_base 축소 시 bottleneck 입력 폭만 변경 (신규 초기화, 훈련 대상)
self.dec_base = base_channels if dec_base is None else int(dec_base)
db = self.dec_base                            # dec1/2/3 = db*{4,2,1}
if db != eb:                                  # 인코더/디코더 폭 차이를 1x1로 브리징
    self.skip_adapters = nn.ModuleDict({
        "a3": nn.Conv2d(eb * 4, db * 4, 1),
        "a2": nn.Conv2d(eb * 2, db * 2, 1),
        "a1": nn.Conv2d(eb, db, 1),
    })

# tools/gazer_distill.py — Annealing 동결 정책 (warm-start 체인의 각 변종)
enc_changed = enc_base is not None and int(enc_base) != base
dec_changed = int(dec_base) != base
train_names = set()
if dec_changed:  train_names |= set(DEC_STAGES)     # up/dec/final_cls(db)
if enc_changed:  train_names |= set(ENC_STAGES)     # enc1/2/3(eb)
                 train_names.add("bottleneck")      # 입력 폭이 enc_base로 변경
if bb.skip_adapters is not None: train_names.add("skip_adapters")
# temporal_mamba + gazer head: 항상 동결 (RL 정책의 KL anchor)
loss = DistillLoss(...)                       # KL(p_s||p_t) + alpha_tv(epoch)*TV
#   alpha_tv: 0.001 -> 0.05 annealing (configs/gazer_distill_t5.yaml)
# warm-start 체인: 각 변종은 이전 변종 best ckpt에서 shape-aware 시작
# --gate: val 전용 SAM3 landscape 위에서 policy+모든 변종을 동일 지표로 재측정
#   delta = student - teacher >= -0.01 통과 변종 중 params 최소 = 선정
"""

CODE_DISTILL_CFG = """# configs/gazer_distill_t5.yaml (주요 설정)
grid:                  # 2x2 annealing 그리드 (enc_base, dec_base) — base=32
  - [32, 16]           # e32d16: decoder 반 폭 (warm-start 체인 1단계)
  - [32, 8]            # e32d8:  decoder 1/4 폭
  - [16, 16]           # e16d16: encoder 반 폭 + decoder 반 폭
  - [16, 8]            # e16d8:  양쪽 1/4 폭 (annealing 완주 변종)
gate: 0.01             # RL 보상 게이트: student agreement >= teacher - 0.01
                       #   통과 변종 중 params 최소 = 선정
epochs: 30             # 런칭 시 --epochs 15 (per-step 벤치 0.258s -> 10.2 h/epoch)
lr: 0.0005  weight_decay: 0.0001  max_grad_norm: 1.0  val_every: 3
tv_anneal_from: 0.001  tv_anneal_to: 0.05             # annealing regularizer
"""


# ---------------------------------------------------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def parse_val_history():
    """ep -> val/agreement_teacher from the training log (last run only is
    fine; the log appends and the current run restarted from ep1)."""
    vals = {}
    if not os.path.exists(TRAIN_LOG):
        return vals
    pat = re.compile(
        r"ep(\d+) \[grpo\] VAL val/agreement_teacher=([\d.]+)")
    with open(TRAIN_LOG, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                vals[int(m.group(1))] = float(m.group(2))
    return vals


def parse_stats_sampled(max_points=60):
    """(ep, it, {metric: value}) sampled from the it-log lines."""
    rows = []
    if not os.path.exists(TRAIN_LOG):
        return rows
    pat = re.compile(r"ep(\d+) \[grpo\] it(\d+)/(\d+) (.*)")
    keypats = {k: re.compile(k + r"=([-\d.eE]+)")
               for k in ("grad/norm", "grpo/kl", "grpo/adv_std",
                         "grpo/neg_pg", "grpo/reward", "grpo/entropy")}
    all_rows = []
    with open(TRAIN_LOG, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            vals = {k: float(p.search(line).group(1))
                    for k, p in keypats.items() if p.search(line)}
            if len(vals) < 4:
                continue
            all_rows.append((int(m.group(1)), int(m.group(2)), vals))
    if not all_rows:
        return rows
    step = max(1, len(all_rows) // max_points)
    return [r for i, r in enumerate(all_rows) if i % step == 0]


def git_info():
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    tags = [t for t in run("git", "tag", "-l").splitlines()
            if t.startswith("v4.0")]
    commits = run("git", "log", "--oneline", "-5")
    return tags, commits


# ---------------------------------------------------------------------------
def svg_line_chart(series, width=560, height=220, yfmt="{:.4f}",
                   title=""):
    """series: list of (name, color, [(x, y), ...]); x = epoch (int)."""
    pad_l, pad_r, pad_t, pad_b = 58, 14, 26, 30
    xs = [p[0] for s in series for p in s[2]]
    ys = [p[1] for s in series for p in s[2]]
    if not xs:
        return f'<div class="chart-empty">{title}: 데이터 없음</div>'
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1e-9
    rng = (y1 - y0) * 0.08
    y0, y1 = y0 - rng, y1 + rng

    def sx(x):
        return pad_l + (x - x0) / max(1e-9, x1 - x0) * (width - pad_l - pad_r)

    def sy(y):
        return pad_t + (1 - (y - y0) / (y1 - y0)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    parts.append(f'<text x="{width/2}" y="14" class="ct">{title}</text>')
    # y gridlines + labels
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        yy = sy(yv)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" '
                     f'y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" class="ax" '
                     f'text-anchor="end">{yfmt.format(yv)}</text>')
    # x ticks
    for x in range(int(x0), int(x1) + 1):
        xx = sx(x)
        parts.append(f'<text x="{xx:.1f}" y="{height-8}" class="ax" '
                     f'text-anchor="middle">{x}</text>')
    # series
    for name, color, pts in series:
        if len(pts) < 2:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" class="ln" stroke="{color}"/>')
        lx, ly = pts[-1]
        parts.append(f'<circle cx="{sx(lx):.1f}" cy="{sy(ly):.1f}" r="3.2" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{width-pad_r-2}" y="{sy(ly)-6:.1f}" '
                     f'class="ax" text-anchor="end" fill="{color}">'
                     f'{name} {yfmt.format(ly)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bar_chart(items, width=560, height=200, fmt="{:.1f}", unit=""):
    """items: (label, value, color). Horizontal bars, sorted desc by caller."""
    row_h = min(22, max(12, (height - 20) / len(items)))
    height = len(items) * row_h + 16
    label_w = 96
    vmax = max(v for _, v, _ in items) or 1e-9
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for i, (lab, v, color) in enumerate(items):
        y = i * row_h
        w = v / vmax * (width - label_w - 70)
        parts.append(f'<text x="{label_w-6}" y="{y+row_h*0.72:.1f}" '
                     f'class="ax" text-anchor="end">{lab}</text>')
        parts.append(f'<rect x="{label_w}" y="{y+2:.1f}" width="{w:.1f}" '
                     f'height="{row_h-6:.1f}" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{label_w+w+5:.1f}" y="{y+row_h*0.72:.1f}" '
                     f'class="ax">{fmt.format(v)}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
def main():
    bench = load_json(BENCH_JSON)
    evalr = load_json(EVAL_JSON)
    vals = parse_val_history()
    stats = parse_stats_sampled()
    tags, commits = git_info()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- 시연 목표 예산 (사용자 목표: 아이캠 2 스트림 x 120Hz + 월드캠 60Hz
    # 게이즈 매핑, 단일 4090, T=8. 실험 서버는 1 스트림만 측정 => 상대 비율만
    # 유효하고, 예산 충족 여부는 시연 환경 실측으로 판정) -------------------
    if bench:
        _fw = bench["forward_ms"]
        _ms1 = _fw["streaming_B1_fifo_perframe_mamba"]
        _bud = 1000.0 / 240.0          # 2 스트림 x 120Hz => GPU 4.17 ms/프레임
        _rat = _ms1 / _bud
    else:
        _ms1 = _bud = _rat = 0.0

    # ---- step 2: stress test ------------------------------------------
    _have_fl = False
    cnn_t = fl_enc = fl_bot = fl_dec = ""
    if bench:
        fl = bench["flops_T5_B1"]
        fl_items = sorted(
            [(k.replace("temporal_mamba", "mamba3"), fl[k],
              "#7c5cff" if "mamba" in k else "#3f8cff")
             for k in fl if not k.startswith("_")
             and k != "temporal_mamba_bound"],
            key=lambda t: -t[1])
        fl_svg = svg_bar_chart(fl_items, fmt="{:.1f}", unit=" GF")
        cnn_t = fl["_cnn_total"]
        mam_t = fl["_mamba_bound"]
        mt = bench["module_time_ms_B1"]
        enc, mam, dec = (mt["cnn_encoder_448to56_T5"],
                         mt["mamba3_T5"], mt["cnn_decoder_56to448_T5"])
        fl_enc = sum(v for k, v in fl.items() if k.startswith("enc"))
        fl_bot = fl["bottleneck"]
        fl_dec = cnn_t - fl_enc - fl_bot
        _have_fl = True
        tot = enc + mam + dec
        th = bench["throughput"]
        fw = bench["forward_ms"]
        tf = bench["effective_TFLOP_s_B1bf16"]
        stress_html = f"""
<h3>2-1. FLOPs (B=1, T=5, 448×448) — 모듈별</h3>
{fl_svg}
<p class="note">총 {fl['_total']:.1f} GFLOP 중 <b>CNN {cnn_t:.1f} / Mamba3 {mam_t:.1f}
→ CNN : Mamba3 = {cnn_t/mam_t:.1f} : 1</b> (매개변수 2.40M).</p>
<h3>2-2. Wall time (B=1, T=5, bf16) — 단계별 비율</h3>
<ul>
<li>CNN encoder (448→56): <b>{enc:.2f} ms ({enc/tot*100:.0f}%)</b>,
effective {tf['cnn_encoder']:.1f} TFLOP/s</li>
<li>Mamba3 (T=5): <b>{mam:.2f} ms ({mam/tot*100:.0f}%)</b>,
effective {tf['mamba3_T5']:.1f} TFLOP/s — T 스윕: T1 {mt['mamba3_T1']:.2f} /
T5 {mt['mamba3_T5']:.2f} / T8 {mt['mamba3_T8']:.2f} ms (거의 평탄, 고정 오버헤드 지배)</li>
<li>CNN decoder (56→448): <b>{dec:.2f} ms ({dec/tot*100:.0f}%)</b>,
effective {tf['cnn_decoder']:.1f} TFLOP/s</li>
</ul>
<h3>2-3. Throughput (실험 서버 3090, 1 스트림 통제 벤치)</h3>
<table>
<tr><th>모드</th><th>소요/프레임</th><th>FPS</th><th>비고</th></tr>
<tr><td>학습용 (T=5 전체 프레임, B=2, mamba ON)</td><td>{fw['T5_allframes_B2_mamba']:.1f} ms</td><td>—</td><td>{th['train_windows_per_s_B2']:.1f} window/s</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba ON)</td><td>{fw['streaming_B1_fifo_perframe_mamba']:.2f} ms</td><td><b>{th['streaming_fps_mamba']:.1f}</b></td><td>1 스트림 기준 — 시연 목표가 아님</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba OFF)</td><td>{fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms</td><td>{th['streaming_fps_nomamba']:.1f}</td><td>mamba 기여 {fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms/프레임</td></tr>
</table>
<h3>2-4. 시연 목표 예산 분석 (최종 목표 기준)</h3>
<p>시연 최종 목표: <b>단일 RTX 4090에서 T=8</b>, 아이캠 2 스트림 각 120Hz 실시간
세그멘테이션 + 월드캠 60Hz 게이즈 매핑(CPU)이 병목 없이 처리.
GPU 예산 = 1000 ms / (2 스트림 × 120 Hz) = <b>{_bud:.2f} ms/스트림 프레임</b>.
실험 서버 실측(1 스트림) {_ms1:.2f} ms/프레임 → <b>스트림당 예산({_bud:.2f} ms)의
{_rat:.1f}배</b>, 이미 1스트림도 초과. 2스트림 병렬 시 GPU 총 수요 =
2×{_ms1:.2f} = {2*_ms1:.2f} ms/사이클, 사이클 예산(2×{_bud:.2f} = {2*_bud:.2f} ms)의
<b>{_rat:.1f}배 초과</b> (3090 기준, T=5 실측).
과거 실측 기록: <b>FIFO 도입 전 T=3에서 1,000프레임 중 ~800프레임(80%)이
레이턴시 초과로 탈락</b> — 시연 환경이 속도 부족 상태였다는 직접 증거.
이 벤치는 인풋 1 스트림만 넣은 통제 실험이므로, 좋은 FPS 수치를 시연 환경의
절대 지표로 해석해선 안 된다.</p>
<h3>2-5. 판정</h3>
<div class="verdict">가설 <b>확인</b>: 연산 병목은 Mamba3가 아니라 <b>CNN</b>.
FLOPs 21.4:1, wall time의 61%는 <b>CNN decoder</b> (dec1/2/3 각각
{fl['dec1']:.0f} GFLOP — 전체 forward의 1/3). Mamba3는 스트리밍 프레임당
+{fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms만 기여.
이 병목 측정은 annealing <b>상대의 크기를 정량화</b>한 근거이며,
4단계의 annealing 대상은 bench 수치로 CNN 인코더·디코더 <b>양쪽</b>
(합계 {fl_enc + fl_dec:.0f} GF, 전체 forward {fl['_total']:.0f} GF의
{(fl_enc + fl_dec) / fl['_total'] * 100:.1f}% — bottleneck {fl_bot:.0f} GF는
출력 폭이 Mamba3 입력이라 고정). 대상의 <b>최종 선정은
RL 보상 게이트(4단계 Stage B)의 실측 하락 허용도</b>가 판정 —
단계적 fade-out 후 agreement가 teacher−0.01 이상 유지되는 가장 가벼운
변종. 시연 예산은 스트림당 {_bud:.2f} ms(2스트림 병렬 시 {_bud:.2f}
ms/사이클)인데, 풀 모델({_ms1:.2f} ms/프레임)은 스트림당으로도
{_rat:.1f}배 초과 — 1스트림만으로도 3090 기준 이미 초과이므로 여유가
0에 가까움. <b>경량화는 시연 목표 달성을 위한 필수 경로</b>.
4090의 실제 가속 계수는 미측정이므로 4090 풀 모델이 예산을 통과할
여지도 있으나, 이는 <b>시연 환경 실측으로만 판정</b>한다. 실험 서버
수치는 모듈별 상대 비율(인코더:mamba:디코더, 변종 간 개선 폭)만
유효하게 사용한다.</div>
<h3>2-6. 측정 코드 (tools/gazer_bench.py 발췌)</h3>
<pre class="code">{CODE_BENCH}</pre>"""
    else:
        stress_html = "<div class='chart-empty'>bench JSON 없음 — tools/gazer_bench.py 재실행 필요</div>"

    # ---- step 3: diagnosis + fix ---------------------------------------
    if vals:
        val_pts = [(e, v) for e, v in sorted(vals.items())]
        val_svg = svg_line_chart(
            [("val/agreement_teacher", "#2ecc71", val_pts)],
            yfmt="{:.4f}", title="VAL agreement (SAM3 teacher 기준)")
    else:
        val_svg = '<div class="chart-empty">VAL 데이터 없음</div>'

    if stats:
        def ser(key, color, name):
            return (name, color,
                    [(e, v[key]) for e, _, v in stats if key in v])
        stat_svg = svg_line_chart(
            [ser("grad/norm", "#e74c3c", "grad/norm"),
             ser("grpo/kl", "#f39c12", "grpo/kl")],
            yfmt="{:.4f}", title="grad/norm · grpo/kl (ep 진행)")
        adv_svg = svg_line_chart(
            [ser("grpo/adv_std", "#9b59b6", "grpo/adv_std"),
             ser("grpo/reward", "#3498db", "grpo/reward (pixel-mean)")],
            yfmt="{:.4f}", title="adv_std · reward")
    else:
        stat_svg = adv_svg = '<div class="chart-empty">stats 없음</div>'

    # ---- step 3a: eval compare (may be pending) -------------------------
    if evalr:
        # JSON keys: config, time, then one dict per model
        model_rows = []
        for k, v in evalr.items():
            if isinstance(v, dict) and "agreement_teacher" in v:
                model_rows.append(
                    f"<tr><td>{k}</td><td><b>{v['agreement_teacher']:.4f}</b>"
                    f"</td><td>{v['val_pass_s']:.0f} s</td></tr>")
        eval_html = ("<p>동일 지표(SAM3 reward-landscape 기준 val/agreement_teacher) 비교 — "
                     f"측정 시각 {evalr.get('time','?')}:</p>"
                     "<table><tr><th>모델</th><th>agreement</th><th>소요</th></tr>"
                     + "".join(model_rows) + "</table>")
    else:
        eval_html = """<div class="pending">측정 중 — <code>tools/gazer_eval_compare.py</code>
(val-only SAM3 landscape ~2h → 3개 모델 VAL 패스). 완료 시 이 스크립트 재실행하면 자동 반영:</div>
<table><tr><th>모델</th><th>agreement</th></tr>
<tr><td>gazer_ep9 (GRPO 8ep, 10-26 런)</td><td>대기</td></tr>
<tr><td>init_fold1 (semantic 초기화 = ep0)</td><td>대기</td></tr>
<tr><td>baseline_400 (pseudo-label 시퀀스 학습)</td><td>대기</td></tr>
</table>
<p class="note">참고: baseline의 공식 수치 val_dice <b>0.8962</b>는 <i>수도라벨 대비</i>
Dice, GAZER의 0.7778은 <i>SAM3 landscape 대비</i> agreement — 지표가 다르므로
동일 지표 원샷 비교를 시행 중. (per-class: BG 0.9829 / Sclera 0.8947 /
Iris 0.8654 / Pupil 0.8417)</p>"""

    # ---- step 4: lightweighting (executed) -------------------------------
    lite_html = f"""
<h3>4-0. 목적 재확인 (10-27 사용자 정정 반영)</h3>
<div class="verdict">경량화의 목적은 학습 시간 절약이 아니라 <b>인퍼런스
실시간성</b>. 시연 환경 = <b>RTX 4090 1장</b>, 최종 목표 T=8에서
아이캠 2 스트림 × 120Hz + 월드캠 60Hz 게이즈 매핑(CPU) 병목 없음.
실험 서버(3090, 1 스트림) 수치는 상대 비율(모듈 기여도, 변종 간 개선 폭)만
유효하며, 시연 예산({_bud:.2f} ms/스트림 프레임) 충족 여부는
<b>시연 환경 실측</b>으로 판정. 실험 목적 = <b>RL 보상 유지 전제 최대
경량화</b>: <code>val/agreement_teacher</code>가 gazer_ep9(teacher)보다
−0.01 이내를 유지하면서 params/latency를 최대화(최소화)하는 변종 선정.</div>
<h3>4-1. Annealing 대상: CNN 인코더 + 디코더 2×2 그리드 (사용자 정정 반영)</h3>
<p>bench가 CNN = FLOPs {cnn_t:.0f} GF(T5)로 CNN 전체(decoder
{fl_dec:.0f} GF 64.5% + 인코더 {fl_enc:.0f} GF 24.5% +
bottleneck {fl_bot:.0f} GF 8.9%)가 병목 확인. 2단계 정정
(10-27): annealing 대상은 연산 병목 <b>위치</b>가 아니라, 단계적
fade-out 후 성능 유지 또는 합리적 하락을 RL+SAM3 튜닝으로 커버 가능한
<b>연산량이 큰 모듈</b> — bench 수치로 CNN 인코더·디코더 <b>양쪽</b>이
대상(합계 {fl_enc + fl_dec:.0f} GF, forward의
{(fl_enc + fl_dec) / fl['_total'] * 100:.1f}% — bottleneck {fl_bot:.0f} GF는
출력 폭이 Mamba3 입력이라 고정). <b>구조적 제약</b>: bottleneck <b>출력</b> 폭
(base×8=256)은 Mamba3 temporal 입력으로, 학습된 Mamba3 가중치 인수를 위해
고정. encoder 폭을 줄이면 bottleneck <b>입력</b> 폭만 바뀌므로
(신규 초기화 → 학습 대상), <code>enc1/2/3</code>만 좁힐 수 있고,
인코더 skip 폭과 디코더 폭 차이는 1×1 skip adapter가 브리징.</p>
<table>
<tr><th>변종</th><th>enc/dec 폭</th><th>params</th>
<th>enc GF</th><th>bot GF</th><th>dec GF (T5)</th><th>CNN GF 합</th>
<th>decoder wall (ms)</th><th>streaming (ms/프레임)</th></tr>
<tr><td>e32d32 (teacher 기준)</td><td>32/32</td><td>2.402M</td>
<td>74.57</td><td>27.75</td><td>216.11</td><td>318.42</td><td>34.71</td><td>10.79 (92.7 FPS)</td></tr>
<tr><td><b>e32d16</b> (체인 1단계, 학습 중)</td><td>32/16</td><td>1.880M</td>
<td>74.57</td><td>27.75</td><td>58.22</td><td>160.53</td><td>15.63</td><td>9.53 (104.9 FPS)</td></tr>
<tr><td>e32d8</td><td>32/8</td><td>1.725M</td>
<td>74.57</td><td>27.75</td><td>16.65</td><td>118.96</td><td>10.82</td><td>7.31 (136.9 FPS)</td></tr>
<tr><td>e16d16</td><td>16/16</td><td>1.507M</td>
<td>18.79</td><td>23.12</td><td>58.22</td><td>100.13</td><td>21.14</td><td>8.73 (114.6 FPS)</td></tr>
<tr><td><b>e16d8</b> (annealing 완주 변종)</td><td>16/8</td><td>1.360M</td>
<td>18.79</td><td>23.12</td><td>16.65</td><td>58.56 (5.4× 감소)</td><td>12.06</td><td>9.40 (106.4 FPS)</td></tr>
</table>
<p class="note">정적 레이더 실측치 (tools/gazer_distill.py --static, 3090,
T=5, 448², 5변종). params/FLOPs는 contention 무관(정적 계산).
wall/streaming ms는 측정 당시 GPU 점유(100%) 환경이라 4090 대비 <b>상한</b> —
변종 간 상대 개선만 인용. spatial 해상도는 변종 불변(bottleneck 출력
H/8=56², pool 3단계 고정; 폭 축소 시 채널 폭만 변함).
encoder 축소: 74.57→18.79 GF(75% 감소), 풀 CNN 318.42→58.56 GF(5.4×).
decoder FLOPs 13× 감소가 wall ~3.2× 감소로 압축되는 점: db≤8 폭에서
conv가 memory/latency-bound(커널당 연산량이 GPU를 채우지 못해
kernel launch + memory traffic이 지배) → FLOPs-비선형.
e16d8의 streaming(9.40 ms)이 e32d8(7.31 ms)보다 큰 점도 같은
narrow-width 효과가 encoder(16-base)까지 확장된 것이며,
동일 decoder 폭(db=8)의 wall 차이(e16d8 12.06 vs e32d8 10.82 ms)는
100% 점유 contention 노이즈 범위 — 4090 clean 실측으로 재확인.
최종 판단은 시연 4090 실측.</p>
<h3>4-2. 실행 방법: Stage A warm-start 체인 → Stage B RL 보상 게이트</h3>
<p><b>Stage A (SAM3-free annealed distillation, warm-start 체인 = 단계적
capacity fade-out)</b>: 체인 순서 <b>e32d16 → e32d8 → e16d16 →
e16d8</b>. 각 변종은 <b>이전 변종의 best checkpoint에서
shape-aware 로드</b>(첫 변종은 gazer_ep9) → "점차 페이드아웃"을
실행 가능한 warm-start 체인으로 구현. 각 변종 trainable:
decoder stages(+skip adapter, 폭 차이 시) / encoder stages + bottleneck
(enc 폭 변화 시). 손실 = <code>KL(p_s‖p_t) + α_tv(epoch)·TV(logits_s)</code>
(p_t = 동결된 gazer_ep9 포워드가 같은 GPU에서 산출, α_tv: 0.001→0.05
annealing). SAM3를 쓰지 않는 이유: 인플라이트 GRPO가 118GB train
landscape를 RAM에 두고 있고, 251GB RAM에 두 개의 landscape를 둘 수 없기
때문. 진행 지표 = val에서 <code>dice(student softmax, policy
softmax)</code>.</p>
<p><b>Stage B (--gate, RL 보상 게이트)</b>: val 전용 SAM3 landscape
(~16,800 프레임, ~27GB RAM, ~1.5–2h)를 1회 빌드해 <b>동일 landscape·동일
지표</b>로 gazer_ep9와 각 <code>student_e*{{dec}}_best.pth</code>를
재측정. 통과 조건 <code>delta = student − teacher ≥ −0.01</code> —
통과 변종 중 <b>params 최소</b>를 선정(= "RL 보상이 유지되는 가장
가벼운 변종"). 이것이 2단계·3단계의 보상 정의(teacher-likelihood)를
4단계의 검증으로 직접 연결하는 인과 고리.</p>
<h3>4-3. 실행 상태 (10-27 17:10 KST)</h3>
<div class="pending">Stage A 체인 1단계 <b>e32d16</b> 학습 중 (15:55 KST
런칭, GPU0, <code>--epochs 15</code>, ep당 ~10.2 h): trainable 0.232M
(1.880M 중) — decoder/skip adapter만 훈련, encoder/Mamba3/bottleneck
동결 (missing=43 = narrow decoder/skip adapter 신규 파라미터,
unexpected=0). 체인 잔여 변종 <b>e32d8 → e16d16 → e16d8</b> 대기
(각 변종은 이전 best에서 <code>--init-ckpt</code> warm-start).
best checkpoint는 VAL(dice_proxy) 피크에서 저장 — epoch 수로
아니라 게이트가 통과 여부를 판정.</div>
<table>
<tr><th>단계</th><th>검증/측정 방법</th><th>결과(수치)</th><th>결론(인과)</th></tr>
<tr><td>정적 레이더 (5변종)</td><td>params + FLOPs(counter) + decoder/streaming wall (CUDA 이벤트 중앙값)</td>
<td>e16d8: 1.360M / CNN 58.56 GF(5.4× 감소) / decoder 16.65 GF(13×) / streaming 9.40 ms</td>
<td>인코더+디코더 양쪽 축소 시 풀 CNN의 1/5.4 — 병목이 2스트림 예산 향상에 구조적으로 기여</td></tr>
<tr><td>student 초기화 (체인 첫 단계)</td><td>gazer_ep9 가중치 shape-aware 로드 (동일 shape 키만)</td>
<td>missing=43, unexpected=0 (e32d16)</td>
<td>inherit가 intact — 체인 시작점(teacher와 동일 feature)에서 fade-out 진행</td></tr>
<tr><td>Stage A smoke (4변종 grid)</td><td>전 변종 build/load/freeze/train/VAL</td>
<td>e32d16/d8 missing=43, e16d16 missing=68, e16d8 missing=74, unexpected=0 (전 변종)</td>
<td>enc 폭 축소 시 bottleneck(입력 폭 변경)+skip adapter가 신규 파라미터로 정합 — freeze 정책 검증</td></tr>
<tr><td>Stage B 게이트</td><td>val-only SAM3 landscape · 동일 지표 agreement_teacher</td>
<td>대기 (변종 4종 학습 + eval-compare 완료 후 실행 예정)</td>
<td>delta ≥ −0.01 통과 변종 중 params 최소 = 시연 후보 — RL 보상 유지 + 최대 경량화 동시 달성</td></tr>
</table>
<h3>4-4. 설계 코드 (발췌)</h3>
<pre class="code">{CODE_DISTILL}</pre>
<pre class="code">{CODE_DISTILL_CFG}</pre>
<h3>4-5. 수용 기준 (시연 목표와 연결)</h3>
<ol>
<li>RL 보상 게이트: <code>agreement_teacher ≥ gazer_ep9 − 0.01</code> (Stage B, 동일 val landscape) — 통과 변종 중 params 최소</li>
<li>시연 4090 실측: T=8, 아이캠 2 스트림 × 120Hz (GPU 4.17 ms/스트림 프레임 예산) +
월드캠 60Hz 게이즈 매핑(CPU) 병목 없음 — <b>실험 서버 수치는 이 기준의 대체가 아님</b></li>
<li>3a(동일 지표 비교)에서 GAZER 정책이 baseline_400 대비 경쟁력이 확인되면,
게이트 통과한 가장 가벼운 변종을 최종 모델로 확정</li>
</ol>"""

    # ---- step 1: version control ----------------------------------------
    tag_rows = "".join(f"<tr><td><code>{t}</code></td></tr>" for t in tags)
    commit_rows = "".join(
        f"<tr><td><code>{c}</code></td></tr>" for c in commits.splitlines())
    step1_html = f"""
<p>브랜치 <code>backbone</code> → <code>github.com/dutlsl/nnUNet</code> (push 완료)</p>
<table><tr><th>GAZER 태그</th></tr>{tag_rows}</table>
<table><tr><th>최근 커밋</th></tr>{commit_rows}</table>"""

    # ---- diagnosis block -------------------------------------------------
    diag_html = f"""
<h3>3-1. 진단: 신호 소실 (10-26 런, ep1–9 로그 수치)</h3>
<div class="diaggrid">
<div class="kpi"><span class="kv">0.016</span><span class="kl">adv_std (ep2+)
— 설계 플러 <code>adv_eps=0.01</code>에 근접</span></div>
<div class="kpi"><span class="kv">≈1.6e-4</span><span class="kl">그룹 reward
표준편차 (adv_std 역산) — 플러의 1/60</span></div>
<div class="kpi"><span class="kv">~5e-4</span><span class="kl">grad/norm
(ep2+) — ep1 초기 0.17의 1/300</span></div>
<div class="kpi"><span class="kv">0.0014–0.0021</span><span class="kl">
grpo/kl (ep2+) — KL 앵커만 동작, PG≈0</span></div>
</div>
<p><b>원인</b>: semantic fold_1 초기값이 near-deterministic(픽셀 엔트로피
0.014 nats) → G개 롤아웃의 <i>프레임 스칼라</i> reward 차이가 1.6e-4에 불과
→ z-score advantage의 std floor(1e-2)가 실제 변동의 60배 → advantage가
플러 노이즈. §7.2의 per-pixel dense reward가 ~2×10⁵ 픽셀 <i>pixel-mean
평균으로 프레임 1개로 붕괴</i>하면서 "프레임 스칼라보다 좋은 credit
assignment"라는 본질적 장점이 소실된 것이 구조적 원인.</p>
<h3>3-2. 수정 (v4.0.1-gazer-dense, 아키텍처 불변)</h3>
<table>
<tr><th>항목</th><th>Before</th><th>After</th></tr>
<tr><td>reward</td><td>프레임 스칼라 (pixel-mean)</td>
    <td><b>per-pixel dense</b> [G,N,H,W]</td></tr>
<tr><td>advantage</td><td>z-score (std floor 1e-2)</td>
    <td><b>raw group-centering</b> clip(r−mean_g r, ±5) — 확신 픽셀 A=0
    정확, 경계 픽셀만 O(1)</td></tr>
<tr><td>importance ratio</td><td>pixel-mean logp</td>
    <td><b>per-pixel</b> logp (categorical_logp_dense)</td></tr>
<tr><td>τ</td><td>1.0</td><td><b>1.5</b></td></tr>
<tr><td>G</td><td>8</td><td><b>16</b> (forward 공유 → 비용 +0)</td></tr>
<tr><td>KL</td><td colspan="2">per-pixel exact (불변, GRPOLoss.dense에 합침)</td></tr>
</table>
<p class="note">Phase 1 smoke: ALL 6 CHECKS PASS (dense grpo_step×2+옵티마이저,
EMA, split 전환, validate, SAM3 miss 폴백, ckpt 저장/로드).</p>
<p><b>인과관계(수치)</b>: ep2+의 adv_std 0.016은 std floor(1e-2)가 실제
변동(1.6e-4)의 60배임을 보여 → advantage가 플러 → grad/norm ~5e-4
(ep1 0.17의 1/300) → policy gradient ≈ 0. dense 전환이 이 구조를
풀면, 확신 픽셀에서 A=0(플러 소멸)이고 경계 픽셀에서만 O(1) 신호가
남아 → ep1에 <b>adv_std ≫ 0.016, neg_pg ≠ 0, grad/norm ~0.17</b>이
되도록 설계된 것이 아래 확인 기준이다.</p>
<h3>3-2b. 수정 코드 (rl/grpo.py · losses/grpo_loss.py 발췌)</h3>
<pre class="code">{CODE_GRPO}</pre>
<h3>3-3. 10-26 런 기록 (신호 소실 전후)</h3>
{val_svg}
{stat_svg}
{adv_svg}
<p class="note">grad/norm·KL이 ep2부터 바닥(플러 노이즈) — 위 진단의 시각화.
VAL은 KL 앵커+미세 신호로 0.7764→0.7778 완만 상승.</p>
<h3>3-4. 동일 지표 베이스라인 비교 (3a)</h3>
{eval_html}
<p class="note"><b>검증 방법(인과관계)</b>: baseline 공식 수치 val_dice 0.8962는
수도라벨(이산 hard label) 대비 Dice이고, GAZER의 0.7778은 SAM3 soft4
landscape 대비 agreement — <i>다른 teacher, 다른 라벨 형식</i>이라 직접 비교 불가.
두 모델을 <b>동일 SAM3 landscape, 동일 dice_agreement 지표</b>로 재측정해야
"RL 세미지도 학습이 수도라벨 시퀀스 학습과 동등/우위인지"를 판단할 수 있다.
val-only landscape(~27GB RAM, ~2h)만 빌드하는 이유는 Sequence 분할의
GT 프레임이 0개라 reward가 SAM3 landscape에서만 오기 때문(전체 landscape의
1/4.4 비용).</p>
<h3>3-4b. 측정 코드 (tools/gazer_eval_compare.py 발췌)</h3>
<pre class="code">{CODE_EVAL}</pre>
<h3>3-5. 재기동 상태</h3>
<p>10-27 12:52 KST: dense 경로 적용 후 자동 순서 오케스트레이션 —
eval-compare(~2h) 종료 → watchdog 재기동(phase0 ~8.7h) → 150 ep.
성공 확인 기준: ep1 <code>adv_std ≫ 0.016</code>, <code>neg_pg ≠ 0</code>,
grad/norm ~0.17 복귀.</p>"""

    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAZER 진행 리포트 — {now}</title>
<style>
:root {{ --bg:#0f1420; --card:#171e2e; --tx:#dce3f0; --mut:#8494ad;
  --ac:#7c5cff; --ok:#2ecc71; --warn:#f39c12; --bad:#e74c3c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--tx);
  font:15px/1.55 'Segoe UI',system-ui,sans-serif; }}
.wrap {{ max-width:920px; margin:0 auto; padding:28px 18px 60px; }}
h1 {{ font-size:26px; margin:0 0 4px; }}
h2 {{ font-size:20px; margin:34px 0 10px; border-bottom:2px solid var(--ac);
  padding-bottom:6px; }}
h3 {{ font-size:16px; margin:22px 0 8px; color:#b9c6e0; }}
p {{ margin:8px 0; }}
table {{ border-collapse:collapse; margin:10px 0; width:100%; }}
th,td {{ border:1px solid #2a3550; padding:6px 10px; text-align:left;
  font-size:14px; }}
th {{ background:#1d2740; }}
code {{ background:#1d2740; padding:1px 5px; border-radius:4px;
  font-size:13px; }}
.card {{ background:var(--card); border:1px solid #243050; border-radius:10px;
  padding:16px 18px; margin:12px 0; }}
.note {{ color:var(--mut); font-size:13.5px; }}
.chart {{ width:100%; height:auto; }}
.chart .grid {{ stroke:#243050; stroke-width:1; }}
.chart .ax {{ fill:var(--mut); font-size:11px; }}
.chart .ln {{ fill:none; stroke-width:2; }}
.chart .ct {{ fill:#b9c6e0; font-size:12px; font-weight:600;
  text-anchor:middle; }}
.chart-empty {{ color:var(--mut); padding:14px; background:var(--card);
  border-radius:8px; }}
.verdict {{ background:#12271c; border:1px solid var(--ok); border-radius:8px;
  padding:12px 14px; }}
.pending {{ background:#2a2410; border:1px solid var(--warn); border-radius:8px;
  padding:10px 14px; }}
.kpi {{ background:var(--card); border:1px solid #243050; border-radius:8px;
  padding:10px 12px; }}
.kv {{ display:block; font-size:22px; font-weight:700; color:var(--ac); }}
.kl {{ color:var(--mut); font-size:13px; }}
.diaggrid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.code {{ background:#0c111c; border:1px solid #243050; border-radius:8px;
  padding:12px 14px; overflow-x:auto; font:12.5px/1.55
  'SFMono-Regular',Consolas,Menlo,monospace; color:#c8d4ea;
  white-space:pre; margin:8px 0; }}
.badge {{ display:inline-block; font-size:12px; padding:2px 10px;
  border-radius:20px; margin-right:6px; }}
.b-done {{ background:#12271c; color:var(--ok); border:1px solid var(--ok); }}
.b-run {{ background:#2a2410; color:var(--warn); border:1px solid var(--warn); }}
.b-wait {{ background:var(--card); color:var(--mut); border:1px solid var(--mut); }}
</style></head><body><div class="wrap">
<h1>GAZER 진행 리포트</h1>
<p class="note">생성: {now} · AutoGaze × GRPO — OpenEDS2019 label-scarce 시퀀스
분할 · SAM3 live teacher · 정책 = 분할 backbone 그 자체 (plan §7) ·
데이터 소스: gazer_bench.json / gazer_grpo_backbone_t5.log / gazer_eval_compare.json</p>

<h2>연구 목표 (3단계)</h2>
<div class="card">
<table>
<tr><th>#</th><th>목표</th><th>본 리포트 내 대응</th></tr>
<tr><td>1</td><td><b>VLM(SAM3)을 삼아 베이스라인의 수도라벨 의존성 없애기</b> —
semantic fold_1 초기화 + SAM3 soft4 reward-landscape로 pseudo-label 없이
시퀀스 도메인 적응</td>
  <td>§3 (SAM3 teacher + reward landscape) · §3a (동일 지표로 베이스라인
  비교 — 수도라벨 의존 탈피 검증)</td></tr>
<tr><td>2</td><td><b>RL(GRPO/PPO) 튜닝으로 시퀀스 세미지도 학습</b> —
정책 = 분할 backbone, per-pixel mask가 action, dense teacher-likelihood
reward로 on-policy GRPO</td>
  <td>§3-2 (dense per-pixel GRPO 신호 복원) · §3-3 (학습 로그 수치)</td></tr>
<tr><td>3</td><td><b>최종 스테이지: VLM+RL 가중치 + Annealing된 경량 모델</b> —
RL 보상이 유지되도록 연산 병목 모듈을 annealing</td>
  <td>§2 (병목 = CNN decoder 확인 + 시연 2스트림 예산 분석) · §4
  (decoder 채널 annealing 실험 — Stage A distill → RL 보상 게이트 →
  4090 T=8 2스트림 120Hz 실측으로 최종 rung 선정)</td></tr>
</table>
</div>

<h2>실행 계획 (5단계) 현황</h2>
<table>
<tr><th>#</th><th>단계</th><th>상태</th><th>요약</th></tr>
<tr><td>1</td><td>버전관리</td>
  <td><span class="badge b-done">완료</span></td>
  <td>fork(dutlsl) push + GAZER 전용 태그 v4.0.0-gazer / v4.0.1-gazer-dense</td></tr>
<tr><td>2</td><td>스트레스 테스트 (병목 검증)</td>
  <td><span class="badge b-done">완료 — 가설 확인</span></td>
  <td>CNN : Mamba3 FLOPs = 21.4:1, wall 61% = CNN decoder → annealing 대상 크기 정량화
  (CNN 인코더+디코더 = forward의 {(fl_enc + fl_dec) / fl['_total'] * 100:.0f}%).
  시연 목표(T=8, 아이캠 2스트림×120Hz, 4090 1장) 예산 {_bud:.2f} ms/프레임 대비
  1스트림 실측 {_ms1:.2f} ms → 스트림당 예산의 {_rat:.1f}배 초과 (FIFO 전 T=3: 80% 프레임 탈락 기록)</td></tr>
<tr><td>3</td><td>성능 진단 + 최적 파라미터</td>
  <td><span class="badge b-run">진행 중</span></td>
  <td>신호 소실 진단(adv_std 0.016=플러) → per-pixel dense GRPO + τ1.5/G16 적용.
  dense 경로 재학습 phase0 진행 중, ep1 신호 확인 대기(adv_std≫0.016, neg_pg≠0)</td></tr>
<tr><td>4</td><td>경량화 + Annealing (실험 실행)</td>
  <td><span class="badge b-run">실행 중</span></td>
  <td>CNN 인코더+디코더 2×2 그리드 annealing (e32d16→e32d8→e16d16→e16d8,
  warm-start 체인, 정적 레이더: e16d8 = 1.360M / CNN FLOPs 5.4× 감소) —
  Stage A 체인 1단계(e32d16) 학습 중, RL 보상 게이트(agreement_teacher ≥
  teacher−0.01, 통과 변종 중 params 최소 선정) + 4090 T=8 2스트림 120Hz
  실측으로 최종 변종 확정</td></tr>
<tr><td>5</td><td>HTML 리포트</td>
  <td><span class="badge b-done">본 문서</span></td>
  <td><code>tools/gazer_report.py</code>로 재생성 (eval 결과 반영 가능)</td></tr>
</table>

<h2>1. 버전관리</h2>
<div class="card">{step1_html}</div>

<h2>2. 스트레스 테스트 — 병목 원상 확인</h2>
<div class="card">{stress_html}</div>

<h2>3. 성능 진단·수정 + 동일 지표 비교</h2>
<div class="card">{diag_html}</div>

<h2>4. 경량화 + Annealing (실험 실행 중)</h2>
<div class="card">{lite_html}</div>

<h2>5. 리포트 재생성</h2>
<div class="card"><p>
<code>nnunet_env/bin/python tools/gazer_report.py</code> —
bench/eval-compare JSON과 학습 로그를 읽어 본 문서를 재생성한다.
eval-compare(JSON) 완료 후 1회 재실행하면 3a 표와 4단계 개문 판정이 자동 반영된다.</p></div>
</div></body></html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html_doc)
    print(f"[report] wrote {OUT} "
          f"(bench={'yes' if bench else 'no'}, "
          f"eval={'yes' if evalr else 'PENDING'}, "
          f"val_points={len(vals)}, stats_rows={len(stats)})")


if __name__ == "__main__":
    main()
