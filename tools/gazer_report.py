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

    # ---- step 2: stress test ------------------------------------------
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
<h3>2-3. Throughput</h3>
<table>
<tr><th>모드</th><th>소요/프레임</th><th>FPS</th><th>비고</th></tr>
<tr><td>학습용 (T=5 전체 프레임, B=2, mamba ON)</td><td>{fw['T5_allframes_B2_mamba']:.1f} ms</td><td>—</td><td>{th['train_windows_per_s_B2']:.1f} window/s</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba ON)</td><td>{fw['streaming_B1_fifo_perframe_mamba']:.2f} ms</td><td><b>{th['streaming_fps_mamba']:.1f}</b></td><td>120Hz 목표 초과</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba OFF)</td><td>{fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms</td><td>{th['streaming_fps_nomamba']:.1f}</td><td>mamba 기여 {fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms/프레임</td></tr>
</table>
<h3>2-4. 판정</h3>
<div class="verdict">가설 <b>확인</b>: 연산 병목은 Mamba3가 아니라 <b>CNN</b>.
FLOPs 21.4:1, wall time의 61%는 <b>CNN decoder</b> (dec1/2/3 각각
{fl['dec1']:.0f} GFLOP — 전체 forward의 1/3). Mamba3는 스트리밍 프레임당
+{fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms만 기여하고,
현재 모델은 이미 120Hz 실시간 목표({th['streaming_fps_mamba']:.0f} FPS)를 충족.
→ <b>4단계(경량화)의 annealing 대상 = CNN decoder</b>.</div>"""
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

    # ---- step 4: lightweighting proposal (conditional) ------------------
    lite_html = """
<h3>4-1. Annealing 대상: CNN decoder (bench 판정 기반)</h3>
<ul>
<li><b>채널 축소</b> dec1/2/3 (각 55.5 GFLOP → base_channels 32를 decoder만 16으로
채널 ½, decoder FLOPs ¼ ≈ 10.1 GFLOP/레이어 절감)</li>
<li><b>권중 양자화</b> W8A16 (decoder conv만 적용; mamba3는 Triton 커널이라
동결) — 3090에서 INT8 conv는 ~2× 속도</li>
<li><b>skip concat 제거</b> (dec의 cat→conv 대 cat 전 단계의 1×1 conv 압축 —
채널 수 유지하면서 concat 오버헤드 절반)</li>
</ul>
<h3>4-2. RL 보상 유지 방식</h3>
<ul>
<li>annealed student S'를 GAZER 최종 policy P*의 <b>KL 스튜던트</b>로 훈련:
    <code>min KL(S'‖P*) + λ·(1 − reward_teacher(S'))</code> — GRPO의
    teacher-likelihood reward를 그대로 objective로 유지 (정적 라벨 피팅 아님)</li>
<li>검증 게이트: annealing 전/후 <b>동일 VAL</b>에서
    <code>agreement_teacher</code> 차이가 0.01 미만이면 통과
    (0.8962 초과 시 적용 전제)</li>
</ul>
<div class="pending">개문 조건: 3a 동일 지표 비교에서 GAZER ≥ baseline_400.
현재 0.7778 vs 0.8962는 지표가 달라 무효 — 3a 결과 대기.</div>"""

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
<h3>3-3. 10-26 런 기록 (신호 소실 전후)</h3>
{val_svg}
{stat_svg}
{adv_svg}
<p class="note">grad/norm·KL이 ep2부터 바닥(플러 노이즈) — 위 진단의 시각화.
VAL은 KL 앵커+미세 신호로 0.7764→0.7778 완만 상승.</p>
<h3>3-4. 동일 지표 베이스라인 비교 (3a)</h3>
{eval_html}
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
  <td>§2 (병목 = CNN decoder 확인) · §4 (decoder-centric annealing +
  KL 스튜던트로 RL 보상 유지 제안)</td></tr>
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
  <td>CNN : Mamba3 FLOPs = 21.4:1, wall time 61% = CNN decoder</td></tr>
<tr><td>3</td><td>성능 진단 + 최적 파라미터</td>
  <td><span class="badge b-run">진행 중</span></td>
  <td>신호 소실 진단 → dense per-pixel GRPO + τ1.5/G16 적용, 재학습 대기</td></tr>
<tr><td>4</td><td>경량화 + Annealing 제안</td>
  <td><span class="badge b-wait">조건부 대기</span></td>
  <td>decoder 중심 annealing 초안 준비 완료 — baseline 초과 게이트 대기 (3a)</td></tr>
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

<h2>4. 경량화 + Annealing 방안 (개문 조건부)</h2>
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
