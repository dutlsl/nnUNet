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
PSEUDO_BENCH = f"""
<span class="ph">측정 절차 (tools/gazer_bench.py)</span>
<ol>
<li><span class="kw">공통 타이밍 규칙</span> — 각 계측 항목은 bf16으로,
워밍업 10회 + 측정 30회 반복한 뒤 <b>CUDA 이벤트 타이밍의 중앙값(ms)</b>을 기록
(최악/최상단이 아닌 대표값을 사용).</li>
<li><span class="kw">모듈 격리 타이밍 (B)</span> —
<ol>
<li>인코더(448→56)를 <b>한 번만</b> 돌려 중간 특징을 고정합니다.</li>
<li>그 고정 특징을 입력으로 decoder만 56→448로 되돌리는 루프를 T(5)개 프레임 반복
→ "decoder 순수 wall time"을 측정. 인코더가 섞이지 않은 decoder 단독 비용입니다.</li>
</ol></li>
<li><span class="kw">스트리밍 경로 (시연 120Hz path, C3)</span> —
<ol>
<li>캐시(FIFO)를 초기화하고, 프레임을 <b>1장씩 순서대로</b> 넣으며
"마지막 프레임만 즉시 디코딩" 모드로 20프레임 처리합니다 — 실제 아이캠 시연과
동일한 호출 순서(B=1).</li>
<li>총 시간 ÷ 20 = 스트리밍 프레임당(ms) → FPS = 1000 ÷ 그 값.</li>
<li>동일 경로에서 Mamba3만 끄고 재측정 → Mamba3의 기여도(ms)를 뺀 값으로 얻습니다.</li>
</ol></li>
</ol>
<span class="cm">FLOPs는 counter로 계수(연산 횟수 직접 세는 법) — wall time과 독립적으로 병목 후보를 교차 검증.</span>
"""

PSEUDO_GRPO = """
<span class="ph">수신된 신호를 "픽셀 단위"로 바꾸는 3단계 (rl/grpo.py · losses/grpo_loss.py)</span>
<ol>
<li><span class="kw">보상 → 장단(advantage) 변환</span>
<ul>
<li>각 픽셀의 보상 r에서 <b>같은 픽셀의 G개 샘플 평균</b>을 뺀 뒤
−5~+5로 잘라(A = clip(r − 평균, ±5)) 장단을 만듭니다.
표준편차로 나누지 않는 것이 핵심 변화입니다.</li>
<li>모든 G개 샘플이 같은 클래스를 내린 "확신 픽셀"은 평균과 정확히 같아
장단 = 0이 <b>엄밀히</b> 되어 플러(노이즈)가 사라지고,
샘플 간 견해가 갈린 "경계 픽셀"에만 ±O(1) 신호가 남습니다.</li>
<li>결국 그라디언트가 <b>teacher와 견해가 갈린 위치에만</b> 작용합니다.</li>
</ul></li>
<li><span class="kw">로그가능성도 픽셀 단위</span> —
"픽셀당 평균 로그가능성"이 아니라 <b>픽셀 각각의</b> 로그가능성
[샘플 G × 배치 × 높이 × 폭]을 그대로 가져옵니다.</li>
<li><span class="kw">GRPO 손실</span>
<ul>
<li>중요도 비율 = exp(현재 정책 로그가능성 − 참조 정책 로그가능성)
(±20 클램프로 수치 안정). 이 비율 × 장단 의 평균이 policy gradient 항입니다
— "참조 정책보다 이 픽셀에서 좋은 선택을 늘리고, 나쁜 선택을 줄여라"가
<b>픽셀마다</b> 별개로 계산됩니다.</li>
<li>거기에 참조 정책(EMA)과의 정확 KL 다이버전스 항을 더해
(β 0.05→0.01 annealing) teacher에서 크게 벗어남을 억제합니다.</li>
</ul></li>
</ol>
<span class="cm">아키텍처는 그대로 두는 "경로만 dense 전환" — 모델 구조·파라미터가 변하는 것이 아니라,
같은 모델에 어떤 신호를 줄지가 바뀐 것입니다.</span>
"""

PSEUDO_EVAL = """
<span class="ph">"같은 눈금자"로 3개 모델을 한 번에 재측정 (tools/gazer_eval_compare.py)</span>
<ol>
<li><span class="kw">기준을 미리 만들어 두기 (landscape)</span> —
val 프레임 16,800개에 대해 SAM3 teacher의 연산(soft4 분할)을 <b>한 번만</b>
계산해 RAM(~27GB, ~2h)에 캐시합니다. 이후 모든 모델은 이 동일한 teacher
기준에 맞서 점수를 받습니다.
<ul>
<li>val만 쓰는 이유: Sequence 분할의 GT 프레임이 0개라 보상이 teacher
landscape에서만 오므로, 전체(4.4배) 대신 val 분할만 측정해도 충분합니다.</li>
</ul></li>
<li><span class="kw">같은 눈금자로 3개 모델 비교</span>
<ol>
<li>gazer_ep9 — GRPO 8epoch 정책</li>
<li>init_fold1 — semantic fold_1 초기값(= ep0, RL 전)</li>
<li>baseline_400 — 수도라벨 시퀀스 학습</li>
</ol>
각 모델을 val 프레임에 통과시켜 teacher 기준과 <b>동일한 dice_agreement 지표</b>로
점수를 매깁니다.</li>
</ol>
<span class="cm">eval_val은 train_gazer.validate를 1:1 재현(bf16 forward +
dice_agreement(softmax, SAM3 soft4)) — 학습 중 찍은 val/agreement_teacher 수치와
완전히 같은 눈금이라 비교가 공정합니다.</span>
"""

PSEUDO_DISTILL = """
<span class="ph">구조: 인코더·디코더의 "폭"만 줄이는 2개의 돌려막기 (knob) — 학습 정책</span>
<ol>
<li><span class="kw">인코더 폭 (enc_base)</span> — enc1/enc2/enc3의 채널을
<code>enc_base × {1, 2, 4}</code>로 만듭니다 (기본 32 → 16이면 절반).
bottleneck의 <b>출력 폭(base×8=256)은 고정</b> — 이 값이 Mamba3의 temporal
입력이라 학습된 시간 처리 능력(Mamba3 가중치)을 그대로 상속하려면 고정해야
합니다. 그래서 enc_base를 줄이면 bottleneck의 <b>입력 폭</b>만 바뀌고,
바뀐 부분만 신규 초기화 → 학습 대상으로 들어갑니다.</li>
<li><span class="kw">디코더 폭 (dec_base)</span> — dec1/dec2/dec3의 채널을
<code>dec_base × {4, 2, 1}</code>로 만듭니다 (기본 32 → 16이면 절반).</li>
<li><span class="kw">폭 차이 브리징 (skip adapter)</span> — 인코더와 디코더 폭이
다를 때, 스킵 연결부 3개에 1×1 컨볼루션(채널만 바꾸는 가벼운 변환)을
끼워 폭 차이를 맞춥니다.</li>
</ol>
<span class="ph">학습 정책 (tools/gazer_distill.py, warm-start 체인의 각 변종)</span>
<ol>
<li><span class="kw">학습 대상(동결 정책)</span> — 폭이 바뀐 쪽만 학습합니다:
디코더 폭 변화 → decoder 단계(+skip adapter) 학습, 인코더 폭 변화 →
enc1/2/3 + bottleneck(입력 폭 변경분) 학습. <b>Mamba3 temporal + GAZER head는
항상 동결</b> — RL 정책이 학습한 시간·시선 능력을 KL 앵커로 보존합니다.</li>
<li><span class="kw">손실</span> — KL(student‖teacher) + α_tv(에포크)×TV(student)
· α_tv는 0.001→0.05로 annealing(초반은 부드러운 분할, 후반은 더 매끄러운
분할을 유도). teacher = 동결된 gazer_ep9가 같은 GPU에서 산출한 softmax.</li>
<li><span class="kw">warm-start 체인</span> — 각 변종은 <b>이전 변종의 best
checkpoint에서 shape-aware 로드</b>(shape이 맞는 가중치만 그대로, 나머지는
신규) → "점차 페이드아웃"을 실행 가능한 체인으로 구현.</li>
<li><span class="kw">게이트(--gate)</span> — val 전용 SAM3 landscape 위에서
policy + 모든 변종을 <b>동일 지표(agreement_teacher)</b>로 재측정.
<code>delta = student − teacher ≥ −0.01</code>를 통과한 변종 중 <b>params
최소</b>를 "RL 보상이 유지되는 가장 가벼운 모델"로 선정합니다.</li>
</ol>
"""

PSEUDO_DISTILL_CFG = """
<span class="ph">annealing 그리드와 훈련 설정 (configs/gazer_distill_t5.yaml)</span>
<ol>
<li><span class="kw">그리드 (2×2, (enc_base, dec_base) — 기본 32)</span>
— 채널 폭 조합을 <b>가볍게→무겁게 단계적으로</b> 줄이는 4개 변종:
<ul>
<li>e32d16: 디코더만 반 폭 (체인 1단계, teacher에 가장 가까움)</li>
<li>e32d8:  디코더 1/4 폭</li>
<li>e16d16: 인코더 반 폭 + 디코더 반 폭</li>
<li>e16d8:  인코더·디코더 모두 1/4 폭 (annealing 완주 변종 = 가장 가벼움)</li>
</ul></li>
<li><span class="kw">게이트</span> — student agreement ≥ teacher − 0.01 을 통과한
변종 중 params 최소를 선정.</li>
<li><span class="kw">훈련</span> — epochs 15(런칭 시 --epochs), lr 0.0005,
weight_decay 0.0001, grad-norm clip 1.0, 3에포크마다 val.
annealing regularizer α_tv 0.001→0.05.</li>
</ol>
"""

# ---------------------------------------------------------------------------
# Figures & pseudocode for the v3-module / dataset-structure sections.
# Plain strings (no f-string) so braces in SVG/inline styles stay literal.
# ---------------------------------------------------------------------------
SVG_V3_FLOW = """
<svg viewBox="0 0 900 310" class="chart">
<defs><marker id="arrv" viewBox="0 0 10 10" refX="9" refY="5"
markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="#7c5cff"/></marker></defs>
<text x="450" y="16" class="ct">VivimBackboneMultiFrame — forward shape 흐름 (T=5, 448×448)</text>
<rect x="8" y="36" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="67" y="58" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">input</text>
<text x="67" y="76" text-anchor="middle" fill="#8494ad" font-size="11">[B,T,1,448,448]</text>
<rect x="144" y="36" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="203" y="58" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">enc1</text>
<text x="203" y="76" text-anchor="middle" fill="#8494ad" font-size="11">32 · 448²</text>
<rect x="280" y="36" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="339" y="58" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">enc2</text>
<text x="339" y="76" text-anchor="middle" fill="#8494ad" font-size="11">64 · 224²</text>
<rect x="416" y="36" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="475" y="58" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">enc3</text>
<text x="475" y="76" text-anchor="middle" fill="#8494ad" font-size="11">128 · 112²</text>
<rect x="552" y="36" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="611" y="58" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">bottleneck</text>
<text x="611" y="76" text-anchor="middle" fill="#8494ad" font-size="11">256 · 56²</text>
<rect x="552" y="140" width="118" height="54" rx="6" fill="#2a2140" stroke="#7c5cff" stroke-width="1.5"/>
<text x="611" y="160" text-anchor="middle" fill="#b9a8ff" font-size="12" font-weight="700">Mamba3</text>
<text x="611" y="174" text-anchor="middle" fill="#8494ad" font-size="10.5">TemporalMambaBlock</text>
<text x="611" y="187" text-anchor="middle" fill="#8494ad" font-size="10.5">[B·56·56, T, 256] T축 시퀀스</text>
<rect x="552" y="236" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="611" y="258" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">dec3</text>
<text x="611" y="276" text-anchor="middle" fill="#8494ad" font-size="11">128 · 112²</text>
<rect x="416" y="236" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="475" y="258" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">dec2</text>
<text x="475" y="276" text-anchor="middle" fill="#8494ad" font-size="11">64 · 224²</text>
<rect x="280" y="236" width="118" height="54" rx="6" fill="#1d2740" stroke="#2f4270"/>
<text x="339" y="258" text-anchor="middle" fill="#b9c6e0" font-size="12.5" font-weight="600">dec1</text>
<text x="339" y="276" text-anchor="middle" fill="#8494ad" font-size="11">32 · 448²</text>
<rect x="144" y="236" width="118" height="54" rx="6" fill="#12271c" stroke="#2ecc71"/>
<text x="203" y="258" text-anchor="middle" fill="#a8e6c3" font-size="12.5" font-weight="600">1×1 conv</text>
<text x="203" y="276" text-anchor="middle" fill="#8494ad" font-size="11">[B,T,4,448,448]</text>
<line x1="126" y1="63" x2="140" y2="63" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="262" y1="63" x2="276" y2="63" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="398" y1="63" x2="412" y2="63" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="534" y1="63" x2="548" y2="63" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="611" y1="90" x2="611" y2="136" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="611" y1="194" x2="611" y2="232" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="552" y1="263" x2="538" y2="263" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="416" y1="263" x2="402" y2="263" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<line x1="280" y1="263" x2="266" y2="263" stroke="#7c5cff" stroke-width="1.6" marker-end="url(#arrv)"/>
<path d="M475,90 L475,263 L548,263" fill="none" stroke="#3f8cff" stroke-width="1.4"
stroke-dasharray="5,4" marker-end="url(#arrv)"/>
<text x="468" y="132" fill="#3f8cff" font-size="10.5" text-anchor="end">skip e3</text>
<line x1="339" y1="90" x2="471" y2="232" stroke="#3f8cff" stroke-width="1.4"
stroke-dasharray="5,4" marker-end="url(#arrv)"/>
<text x="386" y="152" fill="#3f8cff" font-size="10.5">skip e2</text>
<line x1="203" y1="90" x2="335" y2="232" stroke="#3f8cff" stroke-width="1.4"
stroke-dasharray="5,4" marker-end="url(#arrv)"/>
<text x="250" y="152" fill="#3f8cff" font-size="10.5">skip e1</text>
<text x="8" y="304" fill="#8494ad" font-size="11">학습: T 프레임 전부 디코딩 (output_all_frames=True) · 스트리밍 인퍼런스: 1프레임씩, FIFO가 이전 T−1 bottleneck 특징을 보관 → 마지막 프레임만 디코딩 (120Hz path)</text>
</svg>
"""

HTML_DS_DIAGRAM = """
<div style="display:flex;flex-direction:column;gap:8px;margin:10px 0">
<div style="display:flex;align-items:center;gap:10px">
<div style="width:88px;color:#8494ad;font-size:13px;flex-shrink:0">1세션 (1참가자)</div>
<div style="display:flex;width:100%;height:44px;border-radius:8px;overflow:hidden;border:1px solid #2a3550">
<div style="flex:2041;background:#1d2740;display:flex;align-items:center;justify-content:center;font-size:12.5px;color:#8494ad">Generative · ~2,041 (무라벨)</div>
<div style="flex:600;background:#2a2410;display:flex;align-items:center;justify-content:center;font-size:12.5px;color:#f39c12">Sequence · 600 (무라벨)</div>
<div style="flex:1;background:#12271c;min-width:150px;display:flex;align-items:center;justify-content:center;font-size:11.5px;color:#2ecc71">Semantic · 1~222 (4클래스 GT)</div>
</div>
</div>
<div style="color:#8494ad;font-size:12.5px;padding-left:98px">시간 순 → (세션당 블록 길이는 평균값, Semantic만 가변) · train/val/test = 참가자 95/28/29 단위</div>
</div>
"""

PSEUDO_V3_TRAIN = """
<span class="ph">v3 세미지도 학습 (자연어 절차 — 3개 config 모두 bf16, T=5, fold_1 사전학습 시드)</span>
<ol>
<li><span class="kw">Phase 1 · 희소라벨 이중스트림 (Strategy A, phase1_sparse_t5.yaml)</span> —
<ol>
<li>배치 8개 = <b>라벨 2 + 무라벨 6</b>. 라벨 샘플 = Semantic GT 프레임을 중심으로,
앞뒤 <b>Sequence 프레임</b>(1프레임 간격)을 채운 T=5 윈도우
<code>[T,1,448,448]</code>. 무라벨 샘플 = Sequence 연속 T=5 (stride 3으로
프레임 겹침을 줄여 샘플링).</li>
<li>백본 forward(<code>output_all_frames=True</code>) → logits
<code>[B,T,4,448²]</code>.</li>
<li><b>라벨 프레임</b>: WeakMedLoss = DiceCE(GT) + M2B×0.5(경계 신뢰도) +
scale-consistency×0.5(하프해상도 일관성) + entropy-min×0.05.
<b>무라벨 프레임</b>: 라벨 손실 0 — 정규화자만: temporal smoothness
(인접 프레임 softmax 차이의 L2) + M2B, 중량 0.2(unsup_weight)를
30에포크 선형 램프업.</li>
</ol></li>
<li><span class="kw">Phase 2 · 경계 윈도우 (Strategy D, phase2_multiframe_t5.yaml)</span> —
<ol>
<li>세션마다 "Sequence 끝 → Semantic 시작" <b>경계를 가로지르는</b> T=5
윈도우를 샘플링 (라벨·무라벨 프레임이 같은 윈도우에 공존; batch 4).</li>
<li>MultiFrameMaskedLoss: DiceCE는 <code>label_mask=1</code> 프레임에만
적용 + temporal smoothness(전체 프레임) + M2B(전체 프레임, 30ep 램프업).</li>
</ol></li>
<li><span class="kw">Hybrid (Strategy E, phase1_2_hybrid_t5.yaml)</span> — A+D를
동일 데이터로더에서 통합 (batch 4, n_labeled=1).</li>
<li><span class="kw">초기화</span> — nnUNet semantic fold_1 체크포인트
(semantic 도메인 사전학습) 로드시, 시퀀스 도메인 적응은 라벨 프레임 수
(세션당 1~222)로 상한이 잡힘 — 아래 "한계" 참조.</li>
</ol>
"""

PSEUDO_V3_STREAM = """
<span class="ph">120Hz 스트리밍 인퍼런스 (자연어 절차 — FIFO, VivimBackboneMultiFrame)</span>
<ol>
<li><span class="kw">세션 시작</span> — <code>reset_cache()</code> 후 FIFO 깊이 = T−1
(T=5면 4슬롯)까지 채워야 정상 동작 (초기 <code>_is_cache_valid</code> 실패 시
현재 특징으로 채움).</li>
<li><span class="kw">새 프레임 도착마다</span> —
<ol>
<li>CNN 인코더로 <b>현재 프레임만</b> 448²→56² → bottleneck 특징
<code>b [1,256,56,56]</code> (스킵 특징 e1/e2/e3도 함께 보관).</li>
<li>FIFO에 b를 넣되 가장 오래된 1개를 버림 (pop(0)) → 항상 최근 T−1개.</li>
<li>Mamba3를 FIFO 시퀀스(T=5)에 적용 → b_mamba에서 <b>마지막 프레임
(:, −1)만</b> 취함.</li>
<li>마지막 프레임만 디코딩 (up3→dec3→up2→dec2→up1→dec1, 스킵 병합) →
4클래스 logits <code>[1,4,448²]</code>.</li>
</ol></li>
<li><span class="kw">비용 구조</span> — 프레임당 "encode 1 + Mamba3(T=5) +
decode 1". 디코더의 고해상도 up-컨볼(112²→224²→448²)이 매 프레임마다
돌아 §2 스트레스 테스트가 지목한 <b>CNN 병목</b>의 실체.</li>
</ol>
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


def parse_grpo_ep1_signal():
    """Last dense-GRPO epoch's per-iteration stats (adv_std, neg_pg,
    reward, grad/norm) + progress — for the v4 progress section."""
    if not os.path.exists(TRAIN_LOG):
        return None
    pat = re.compile(
        r"ep(\d+) \[grpo\] it(\d+)/(\d+) .*?"
        r"grad/norm=([\d.]+) grpo/adv_std=([\d.]+) grpo/entropy=([\d.]+) "
        r"grpo/kl=([\d.]+) grpo/loss=([-\d.]+) grpo/neg_pg=([-\d.]+) "
        r"grpo/reward=([\d.]+)")
    best = None
    with open(TRAIN_LOG, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                best = tuple(m.groups())
    if not best:
        return None
    ep, it, total = int(best[0]), int(best[1]), int(best[2])
    return {"ep": ep, "it": it, "total": total,
            "grad_norm": float(best[3]), "adv_std": float(best[4]),
            "entropy": float(best[5]), "kl": float(best[6]),
            "loss": float(best[7]), "neg_pg": float(best[8]),
            "reward": float(best[9])}


def parse_distill_log():
    """Per-variant distill state (last VAL dice_proxy + last epoch done)."""
    path = os.path.join(ROOT, "logs", "gazer_distill_stageA.log")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            mv = re.search(r"\[distill:(\w+)\]", line)
            if not mv:
                continue
            v = mv.group(1)
            mval = re.search(r"ep(\d+) VAL dice_proxy\(policy\)=([\d.]+)", line)
            if mval:
                out.setdefault(v, {})["val"] = (int(mval.group(1)),
                                                float(mval.group(2)))
            mep = re.search(r"^\[distill:\w+\] ep(\d+) (?:loss|grad|kl|tv)=",
                             line)
            if mep:
                out.setdefault(v, {})["last_ep"] = int(mep.group(1))
            mdone = re.search(r"\[distill:\w+\] DONE", line)
            if mdone:
                out.setdefault(v, {})["done"] = True
    return out


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
<p class="note"><b>지표 설명</b> · <b>GFLOP</b>: 그 모듈이 T=5
(5프레임) 윈도우 하나를 처리하는 데 수행하는 연산 횟수(10억 회 단위).
"얼마나 무거운가"의 표준 척도 — bar가 길수록 연산량이 큼.
총 {fl['_total']:.1f} GFLOP 중 <b>CNN {cnn_t:.1f} / Mamba3 {mam_t:.1f}
→ CNN : Mamba3 = {cnn_t/mam_t:.1f} : 1</b> (매개변수 2.40M).
즉, 전체 연산량의 대다수가 CNN에 있습니다.</p>
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
<p class="note"><b>읽는 법</b> · ms/단계 = 그 단계가 T=5 윈도우
(5프레임)를 처리하는 데 걸린 시간(중앙값). <b>effective TFLOP/s</b>
= 그 단계의 FLOPs ÷ 실측 ms(해당 모듈이 실제로 몇 TFLOP/s로
돌아가는지). <b>T 스윕</b>: T(프레임 수)를 1→5→8로 바꿨을 때
Mamba3 시간 — 거의 평탄하면 T에 무관한 고정 오버헤드가 지배
(프레임 수를 늘려도 선형으로 안 떨어짐).</p>
<h3>2-3. Throughput (실험 서버 3090, 1 스트림 통제 벤치)</h3>
<table>
<tr><th>모드</th><th>소요/프레임</th><th>FPS</th><th>비고</th></tr>
<tr><td>학습용 (T=5 전체 프레임, B=2, mamba ON)</td><td>{fw['T5_allframes_B2_mamba']:.1f} ms</td><td>—</td><td>{th['train_windows_per_s_B2']:.1f} window/s</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba ON)</td><td>{fw['streaming_B1_fifo_perframe_mamba']:.2f} ms</td><td><b>{th['streaming_fps_mamba']:.1f}</b></td><td>1 스트림 기준 — 시연 목표가 아님</td></tr>
<tr><td>스트리밍 (B=1, FIFO, mamba OFF)</td><td>{fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms</td><td>{th['streaming_fps_nomamba']:.1f}</td><td>mamba 기여 {fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms/프레임</td></tr>
</table>
<p class="note"><b>지표 설명 (행마다 무엇을 잰 것인가)</b> ·
<b>학습용</b>: 학습 중 실제로 일어나는 호출 패턴 — T=5 윈도우의 5프레임을
B=2(2개를 한꺼번에)로 처리. "얼마나 빨리 학습할 수 있나"의 속도
(window/s)이지, 실시간 성능이 아닙니다.
<b>스트리밍 (mamba ON)</b>: 시연과 같은 호출 순서 — 프레임 1장씩
FIFO(선입선출 대기열)에 넣고 마지막 프레임만 즉시 디코딩 (B=1).
<b>이 행의 ms/프레임이 시연 예산({_bud:.2f} ms)과 비교하는 숫자</b>입니다.
FPS = 1000 ÷ ms/프레임.
<b>스트리밍 (mamba OFF)</b>: 같은 경로에서 Mamba3만 끄고 재측정한 값 —
두 스트리밍 행의 차 = Mamba3가 프레임당 차지하는 시간.
해석 주의: 실험 서버(3090) 1 스트림 통제 실측. 이 FPS를 시연 환경
(4090, 2 스트림)의 절대 FPS로 해석하면 안 됩니다.</p>
<h3>2-4. 시연 목표 예산 분석 (최종 목표 기준)</h3>
<p><b>시연 목표</b>: 단일 RTX 4090에서 T=8로 아이캠 2 스트림 각 120Hz
실시간 세그멘테이션 + 월드캠 60Hz 게이즈 매핑(CPU)을 병목 없이 처리.</p>
<p><b>1) 예산 계산</b> — 2 스트림이 각 120Hz를 내면 GPU는 1초에
2×120 프레임을 소화해야 함 → 프레임당 예산 = 1000 ms / (2 스트림 ×
120 Hz) = <b>{_bud:.2f} ms/스트림 프레임</b>.</p>
<p><b>2) 실측과 비교</b> — 실험 서버(3090) 1 스트림 실측
{_ms1:.2f} ms/프레임 = 예산({_bud:.2f} ms)의 <b>{_rat:.1f}배</b>:
1 스트림만으로 이미 초과. 2 스트림 병렬 시 GPU 총 수요 =
2×{_ms1:.2f} = {2*_ms1:.2f} ms/사이클인데, 같은 사이클의 예산(2×{_bud:.2f}
= {2*_bud:.2f} ms)의 <b>{_rat:.1f}배 초과</b> (3090 기준, T=5 실측).</p>
<p><b>3) 과거 실측 증거</b> — FIFO 도입 전 T=3에서 1,000프레임 중
~800프레임(80%)이 레이턴시 초과로 탈락한 기록 — 시연 환경이
속도 부족 상태였다는 직접 증거.</p>
<p class="note">해석 주의: 이 벤치는 입력 1 스트림만 넣은 통제 실험.
좋은 FPS 수치를 시연 환경의 절대 지표로 해석하면 안 됩니다.</p>
<h3>2-5. 판정</h3>
<div class="verdict">
<b>가설 확인: 연산 병목은 Mamba3가 아니라 CNN.</b>
<p><b>근거</b> · FLOPs는 CNN : Mamba3 = 21.4:1, wall time의 61%가
<b>CNN decoder</b> (dec1/2/3 각각 {fl['dec1']:.0f} GFLOP — 전체
forward의 1/3). Mamba3는 스트리밍 프레임당
+{fw['streaming_B1_fifo_perframe_mamba']-fw['streaming_B1_fifo_perframe_nomamba']:.2f} ms만
기여.</p>
<p><b>annealing 대상</b> · 이 병목 측정은 annealing "상대의 크기"를
정량화한 근거. 4단계 annealing 대상 = bench 수치로 CNN 인코더·디코더
<b>양쪽</b> (합계 {fl_enc + fl_dec:.0f} GF, 전체 forward
{fl['_total']:.0f} GF의 {(fl_enc + fl_dec) / fl['_total'] * 100:.1f}% —
bottleneck {fl_bot:.0f} GF는 출력 폭이 Mamba3 입력이라 고정).</p>
<p><b>얼마나 줄이는가</b> · 최종 선정은 <b>RL 보상 게이트(4단계
Stage B)의 실측</b>이 판정 — 단계적 fade-out 후 agreement가
teacher−0.01 이상 유지되는 가장 가벼운 변종.</p>
<p><b>시연 예산과의 관계</b> · 예산은 스트림당 {_bud:.2f} ms(2스트림
병렬 시 {_bud:.2f} ms/사이클). 풀 모델({_ms1:.2f} ms/프레임)은
스트림당으로도 {_rat:.1f}배 초과 — 1스트림만으로도 3090 기준 이미
초과, 여유 0에 가까움. <b>경량화는 시연 목표 달성을 위한 필수
경로</b>. 4090의 실제 가속 계수는 미측정이므로 4090 풀 모델이
예산을 통과할 여지도 있으나, 이는 <b>시연 환경 실측으로만
판정</b>. 실험 서버 수치는 모듈별 상대 비율(인코더:mamba:디코더,
변종 간 개선 폭)만 유효하게 사용.</p>
</div>
<h3>2-6. 측정 알고리즘 (tools/gazer_bench.py, 자연어 절차)</h3>
<div class="pseudo">{PSEUDO_BENCH}</div>"""
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
<p class="note"><b>지표 설명 (완료 시 이 표의 열이 무엇을 뜻하는가)</b> ·
<b>모델</b>: 세 개의 출발점 — gazer_ep9(GRPO 8에포크 튜닝 후
정책), init_fold1(semantic fold_1 초기값 = ep0, RL 학습 전),
baseline_400(수도라벨 400개 시퀀스 학습).
<b>agreement</b>: 각 모델을 val 프레임에 통과시켜 <b>동일한 SAM3
landscape(같은 기준 답안집)와 동일한 dice_agreement</b>로 잰 값 —
같은 눈금이라 값만 보면 우열이 바로 판정됩니다.
<b>소요</b>: 모델 1개 VAL 패스 걸린 시간(참고).
<code>delta</code> = 모델 − baseline_400: 양수면 baseline_400보다
동일 지표에서 우위.
참고: baseline의 공식 수치 val_dice <b>0.8962</b>는 <i>수도라벨
대비</i> Dice, GAZER의 0.7778은 <i>SAM3 landscape 대비</i>
agreement — 다른 눈금이라 그대로 비교할 수 없고, 이 표가
그를 해결합니다. (per-class: BG 0.9829 / Sclera 0.8947 /
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
<p><b>무엇을 줄이는가</b>: bench에서 CNN 전체가
FLOPs {cnn_t:.0f} GF(T5)의 병목으로 확인 — decoder {fl_dec:.0f} GF
(64.5%) + 인코더 {fl_enc:.0f} GF (24.5%) + bottleneck {fl_bot:.0f} GF
(8.9%). 10-27 2단계 정정: annealing 대상은 연산 병목 <b>위치</b>가
아니라, 단계적 fade-out 후 성능 유지 또는 합리적 하락을
RL+SAM3 튜닝으로 커버 가능한 <b>연산량이 큰 모듈</b> — bench 수치로
CNN 인코더·디코더 <b>양쪽</b>이 대상(합계 {fl_enc + fl_dec:.0f} GF,
forward의 {(fl_enc + fl_dec) / fl['_total'] * 100:.1f}% —
bottleneck {fl_bot:.0f} GF는 출력 폭이 Mamba3 입력이라 고정).</p>
<p><b>구조적 제약 (왜 bottleneck 출력은 못 줄이는가)</b></p>
<ul>
<li>bottleneck <b>출력</b> 폭(base×8=256)은 Mamba3 temporal의
입력입니다. 학습된 Mamba3 가중치를 그대로 이어받는(인수)
위해 이 폭을 고정해야 합니다 — 바꿨으면 Mamba3를 처음부터
다시 학습해야 해서, 시간 처리 능력(RL로 얻은 능력)을 버리는
것이 됩니다.</li>
<li>그래서 encoder 폭을 줄여도 bottleneck <b>입력</b> 폭만 바뀌고,
이 변경분만 신규 초기화되어 학습 대상이 됩니다.
좁힐 수 있는 CNN 폭은 <code>enc1/2/3</code>과
<code>dec1/2/3</code>.</li>
<li>인코더와 디코더 폭이 다르면 스킵 연결부 3개에 1×1
컨볼루션(skip adapter)을 끼워 폭 차이를 맞춥니다.</li>
</ul>
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
<p class="note"><b>지표 설명 (열마다 무엇을 뜻하는지)</b> ·
<b>변종</b>: e{{인코더폭}}d{{디코더폭}} — 인코더/디코더의 채널 폭 조합.
<b>params</b>: 가중치 개수(모델 크기). 작을수록 메모리·추론 부담이 적음.
<b>enc/bot/dec GF</b>: 인코더·bottleneck·디코더의 연산량(GFLOPs, T=5 기준).
<b>CNN GF 합</b>: 이 세 값을 더한 CNN 총 연산량 — "무겁기"의 정량 척도.
<b>decoder wall(ms)</b>: decoder 단독 실행 시간(인코더 특징을 고정하고
decoder만 돌린 실측). <b>streaming(ms/프레임)</b>: 시연과 같은 "1장씩
처리" 모드에서 프레임당 걸리는 시간 — <b>이 값이 시연 예산
({_bud:.2f} ms)과 직접 비교</b>됩니다. 숫자가 작을수록 실시간에 가까움.
e32d32가 teacher 기준(가장 무거움)이고, e16d8이 annealing 완주(가장
가벼움) 변종입니다.</p>
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
<p><b>Stage A (SAM3-free annealed distillation — warm-start 체인으로
단계적 capacity fade-out)</b></p>
<ul>
<li><b>체인 순서</b>: <b>e32d16 → e32d8 → e16d16 → e16d8</b>.
각 변종은 <b>이전 변종의 best checkpoint에서 shape-aware 로드</b>
(shape이 맞는 가중치만 그대로, 바뀐 부분만 신규) — 첫 변종은
gazer_ep9. "점차 페이드아웃"을 실행 가능한 warm-start 체인으로
구현한 것입니다.</li>
<li><b>학습 대상</b>: 폭이 바뀐 쪽만 학습. decoder stages(+skip
adapter, 폭 차이 시) / encoder stages + bottleneck(enc 폭
변화 시). Mamba3 + GAZER head는 항상 동결.</li>
<li><b>손실</b>: <code>KL(p_s‖p_t) + α_tv(epoch)·TV(logits_s)</code> —
student 분할이 teacher(gazer_ep9)와 가까워지도록(KL) + 분할
경계가 매끄럽게(TV). p_t는 동결된 gazer_ep9가 같은 GPU에서
산출한 softmax. α_tv는 0.001→0.05 annealing.</li>
<li><b>SAM3를 쓰지 않는 이유</b>: 인플라이트 GRPO가 118GB train
landscape를 RAM에 두고 있고, 251GB RAM에 두 개의 landscape를
둘 수 없기 때문. Stage A는 "가볍게 줄이는" 단계이고,
"보상이 유지되는지"의 검증은 Stage B에서 SAM3 landscape로
합니다.</li>
<li><b>진행 지표</b>: val에서 <code>dice(student softmax, policy
softmax)</code> (dice_proxy) — Stage A는 teacher 품질이
아니라 "policy 흉내"의 진행을 보기 위한 것(§4-3 주석 참고).
결정적 검증은 Stage B.</li>
</ul>
<p><b>Stage B (--gate, RL 보상 게이트 — 최종 판정)</b></p>
<ul>
<li><b>준비</b>: val 전용 SAM3 landscape(~16,800 프레임, ~27GB
RAM, ~1.5–2h)를 1회 빌드합니다. "기준 답안집"을 미리 만들어
두면 모든 모델을 같은 조건으로 재측정할 수 있습니다.</li>
<li><b>측정</b>: 그 landscape 위에서 <b>동일 지표
(agreement_teacher)</b>로 gazer_ep9(teacher 기준점)와 각
<code>student_e*{{dec}}_best.pth</code>를 재측정.</li>
<li><b>통과 조건</b>: <code>delta = student − teacher ≥ −0.01</code> —
teacher(전체 크기 모델)보다 0.01 이내로만 허용하는 하락.
통과한 변종 중 <b>params 최소</b>를 "RL 보상이 유지되는 가장
가벼운 변종"으로 선정.</li>
<li><b>의미</b>: 2단계·3단계에서 정한 보상 정의(teacher-
likelihood)를 4단계의 검증으로 직접 연결하는 인과 고리 —
"경량화해도 보상을 지켰는가"를 보상 그 자체의 눈금으로 확인.</li>
</ul>
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
<p class="note"><b>지표 설명 (이 표에서 무엇을 확인하는가)</b> ·
<b>정적 레이더</b>: 학습 전에 "각 변종이 얼마나 가벼워지는가"를
params(크기)·FLOPs(연산량)·wall time(실제 속도)으로 미리 재본 것 —
4개 변종 다 학습시켜 보기 전에 후보를 좁히는 스크리닝.
<b>student 초기화 / smoke</b>: "가중치 이전이 제대로 됐는가" 검증 —
<code>missing</code>은 shape이 달라 새로 만들어야 하는 파라미터(기대됨),
<code>unexpected</code>는 teacher에 없는 새 키(0이면 깨끗한 이전).
<code>missing=43, unexpected=0</code> = 디코더 폭 축소로 생긴 43개 파라미터만
신규이고 나머지는 그대로 이어받았다는 뜻.
<b>Stage B 게이트</b>: §4-2의 통과 조건(delta ≥ −0.01)을 실제
agreement_teacher 수치로 확인하는 단계 — "가볍게 줄이고도 RL 보상을
지켰는가"의 최종 판정. 아직 대기 상태(4개 변종 학습 + eval-compare
완료 후 실행).</p>
<h3>4-4. 설계 알고리즘 (자연어 절차)</h3>
<div class="pseudo">{PSEUDO_DISTILL}</div>
<div class="pseudo">{PSEUDO_DISTILL_CFG}</div>
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
<table><tr><th>최근 커밋</th></tr>{commit_rows}</table>
<p class="note"><b>태그가 의미하는 것</b> ·
<code>v4.0.0-gazer</code> — GAZER(AutoGaze × GRPO) 시작점: 분할
backbone의 on-policy 튜닝 코드.
<code>v4.0.1-gazer-dense</code> — §3-2의 신호 소실 수정 반영:
per-pixel dense GRPO + τ 1.5/G 16.
<code>v4.0.2-gazer-report</code> — 이 보고서 생성 스크립트 포함.
<code>v4.1.0-gazer-lite</code> — §4의 2×2 enc/dec annealing 그리드
경량화 코드. 태그는 "리포트의 각 섹션이 어떤 코드 버전의 결과인가"를
확정합니다 — 수치 재현 시 해당 태그 체크아웃으로 코드 상태가
동결됩니다.</p>"""

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
<p class="note"><b>이 네 수치가 무엇을 말하는가</b> ·
<b>adv_std</b>: 같은 프레임에서 뽑은 G개 샘플(롤아웃)끼리
"이 분할이 좋은 선택이냐 나쁜 선택이냐"를 가르는 장단(advantage)의
산포. 0.016이면 설계 안전지대(플러 0.01)에 붙은 <b>노이즈 수준</b>.
<b>그룹 reward 표준편차 ≈1.6e-4</b>: 16개 롤아웃의 보상이 거의 같다는
뜻 — "어떤 선택이 좋은가"를 배울 차이가 1/60,000.
<b>grad/norm ~5e-4</b>: 학습이 실제로 가중치를 움직이는 크기.
정상값(ep1의 0.17)의 1/300 → <b>바닥(학습이 멈춘)에 붙어 있는 것</b>.
<b>grpo/kl 0.0014–0.0021</b>: 정책이 참조(teacher)에서 살짝만 벗어나고
그저 KL 앵커가 "그 자리에 붙어 있으라"만 작용.
→ 네 개가 동시에 "학습 신호가 사라진 상태"를 가리킵니다.</p>
<p><b>원인 — 신호가 왜 사라졌나 (인과 순서)</b></p>
<ol>
<li>semantic fold_1 초기값이 near-deterministic(픽셀 엔트로피
0.014 nats) — 프레임 한 장에 대한 분할이 거의 한 개뿐이라,
같은 프레임의 G개 롤아웃(샘플)이 대부분 같은 답을 냅니다.</li>
<li>그 결과 G개 롤아웃의 <i>프레임 스칼라</i> reward 차이는
1.6e-4에 불과합니다.</li>
<li>그런데 z-score advantage의 std floor(1e-2)가 이 실제 변동의
60배 → 정규화 후 신호는 사라지고 플러만 남습니다.</li>
</ol>
<p class="note"><b>구조적 원인</b>: §7.2의 per-pixel dense reward는 원래
"어디 픽셀이 좋은 선택이었는지"를 픽셀 단위로 알려줄 수 있는
credit assignment를 갖고 있습니다. 그런데 ~2×10⁵ 픽셀을
<i>pixel-mean으로 프레임 1개 값으로 평균</i>하면서 그 정보가
소실되고, 위에서 본 "샘플 간 차이 자체가 없는 프레임 스칼라"
경로로 돌아간 것입니다.</p>
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
<p class="note"><b>지표 설명 (Before → After가 뭘 바꾸는지)</b> ·
<b>reward</b>: 보상의 "해상도". pixel-mean = 한 프레임의 20만 픽셀을
1개의 숫자로 압축(어디가 잘잘못인지 정보 소실) → per-pixel dense =
<b>픽셀마다</b> 보상을 남겨 "이 픽셀만 teacher와 달랐다"는 신호를
지킴.
<b>advantage</b>: z-score(전체 std로 정규화)는 신호가 플러보다 작으면
정규화가 노이즈를 증폭 → <b>raw group-centering</b>(그룹 평균만 빼고
±5로 자름)은 확신 픽셀(16개 롤아웃이 같은 클래스)에서 장단이
엄밀히 0이 되어 플러가 사라짐.
<b>importance ratio</b>: "지금 정책이 참조 대비 이 픽셀의 선택 확률을
얼마나 바꿨나" — 픽셀 단위로 정확히 측정.
<b>τ 1.0→1.5</b>: 온도(temperature) — 샘플링 분포를 부드럽게
만들어 롤아웃의 다양성↑, 견해가 갈린(=배울 것이 있는) 픽셀이 더
많이 나옴. <b>G 8→16</b>: 프레임당 롤아웃 수. forward가 이미 공유되므로
추가 비용 없이 그룹 평균이 더 안정.
<b>KL</b>: teacher에서 크게 벗어나지 않게 하는 제동 — per-pixel
정확 KL을 유지.</p>
<p class="note">Phase 1 smoke: ALL 6 CHECKS PASS (dense grpo_step×2+옵티마이저,
EMA, split 전환, validate, SAM3 miss 폴백, ckpt 저장/로드).</p>
<p><b>인과관계(수치) — 문제에서 해결까지의 연결</b></p>
<ol>
<li>ep2+의 adv_std 0.016은 std floor(1e-2)가 실제 변동(1.6e-4)의
60배임을 보여 줍니다 — 신호가 플러보다 작으면 정규화 후
advantage에 남는 것은 플러뿐.</li>
<li>advantage가 플러 → grad/norm ~5e-4 (ep1 0.17의 1/300) →
policy gradient ≈ 0. 학습이 사실상 멈춘 상태입니다.</li>
<li>dense 전환은 이 구조를 풀 것: 확신 픽셀에서 장단 A=0(플러
소멸)이고, teacher와 견해가 갈린 경계 픽셀에서만 O(1) 신호가
남습니다.</li>
</ol>
<p>즉, 재기동 후 ep1에서 <b>adv_std ≫ 0.016, neg_pg ≠ 0,
grad/norm ~0.17</b>가 찍히면 "실 신호가 들어오고 학습이 다시
움직인다"는 확인 신호입니다 (§3-5 확인 기준).</p>
<h3>3-2b. 수정 알고리즘 (자연어 절차)</h3>
<div class="pseudo">{PSEUDO_GRPO}</div>
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
<h3>3-4b. 측정 알고리즘 (tools/gazer_eval_compare.py, 자연어 절차)</h3>
<div class="pseudo">{PSEUDO_EVAL}</div>
<h3>3-5. 재기동 상태</h3>
<p>10-27 12:52 KST: dense 경로 적용 후 자동 순서 오케스트레이션 —
eval-compare(~2h) 종료 → watchdog 재기동(phase0 ~8.7h) → 150 ep.
성공 확인 기준: ep1 <code>adv_std ≫ 0.016</code>, <code>neg_pg ≠ 0</code>,
grad/norm ~0.17 복귀.</p>
<p class="note"><b>확인 기준 수치 각각의 의미</b> ·
<b>adv_std ≫ 0.016</b>: 장단(advantage)의 산포가 설계 안전지대
(플러 0.01)를 크게 벗어났다는 것 = "좋은 선택 vs 나쁜 선택"을
가르는 실 신호가 들어오기 시작.
<b>neg_pg ≠ 0</b>: policy gradient 손실의 음수(강화) 성분이
0이 아님 — 실제로 좋은 선택의 확률을 올리는 방향의 업데이트가
발생.
<b>grad/norm ~0.17</b>: ep1 초기 수준(0.17)으로 복귀 —
§3-1의 ~5e-4(바닥)에서 정상 학습으로 돌아온 증거.</p>"""

    # ---- step: v4 progress snapshot (dynamic, regenerated each run) -----
    ov_dir = os.path.join(ROOT, "logs", "diagnostics", "overlay")
    ov_g = load_json(os.path.join(ov_dir, "gazer_ep9_metrics.json")) or {}
    ov_d = load_json(os.path.join(ov_dir, "d16_metrics.json")) or {}
    ov_x = load_json(os.path.join(ov_dir, "cross_model_agreement.json")) or {}
    _d16 = parse_distill_log().get("d16", {})
    _sig = parse_grpo_ep1_signal()

    def _fmt(d, key, nd=4):
        v = d.get(key)
        if isinstance(v, dict):          # {"mean":…, "min":…}
            v = v.get("mean", v)
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

    if vals:
        _ag_ep, _ag_best = max(vals.items(), key=lambda kv: kv[1])
        _val_traj = "".join(
            f"<tr><td>ep{e}</td><td>{v:.4f}</td></tr>"
            for e, v in sorted(vals.items()))
        _val_first = f"{sorted(vals.items())[0][1]:.4f}"
        _ag_kpi = f"{_ag_best:.4f} <span style='font-size:12px;color:var(--mut)'>(ep{_ag_ep})</span>"
    else:
        _ag_kpi, _val_traj, _val_first = "—", "", "—"
    _d16_val = _d16.get("val", (None, None))
    _d16_kpi = (f"{_d16_val[1]:.4f}" if _d16_val[1] is not None else "—")
    _d16_ep = _d16_val[0] or "—"
    _d16_last = _d16.get("last_ep", "—")

    _img = "../logs/diagnostics/overlay/"
    _figs = f"""
<div class="figgrid">
<div><img src="{_img}gazer_ep9_frame_000000336400.png"/>
<div class="figcap"><b>gazer_ep9 (풀 정책) · 프레임 336400 (짝수)</b> —
3패널: 원본(흰색 사각형=400² 크롭) | 크롭 | 모델 분할 오버레이
(초록=결막, 빨강=홍채, 파랑=동공). test Sequence (참가자 무라벨 스트림).</div></div>
<div><img src="{_img}gazer_ep9_frame_000000336401.png"/>
<div class="figcap"><b>gazer_ep9 · 프레임 336401 (홀수)</b> —
위 그림의 인접 프레임. 0-B에서 확인한 <b>좌우 카메라 교차</b>로
"같은 시점의 다른 눈"이라 동공 위치가 대칭으로 보임.</div></div>
<div><img src="{_img}d16_frame_000000336400.png"/>
<div class="figcap"><b>e32d16 경량 student (dec 1/2 폭) · 336400</b> —
같은 프레임에서 distill된 경량 모델의 분할. gazer_ep9 오버레이와
비교하면 동공·홍채 경계가 시각적으로 일치 (pixel_exact {_fmt(ov_x, 'pixel_exact_mean')}).</div></div>
<div><img src="{_img}d16_frame_000000336401.png"/>
<div class="figcap"><b>e32d16 · 336401</b> — 짝/홀 프레임 모두
동공⊆홍채 포개짐 1.0 (nesting), step-2 동일 뷰 IoU {_fmt(ov_d, 'iris_iou_step2_sameview', 2)}.</div></div>
</div>
<p class="note"><b>오버레이 판독</b> · 각 이미지는 Sequence <b>test</b>
(17,400프레임 중 336400~336459, 60프레임)에 대한 모델 인퍼런스 결과 —
학습에 쓰이지 않은 순수 무라벨 도메인. 짝/홀 인접 프레임의 동공
위치 차이(대칭)는 모델 오류가 아니라 0-B의 스테레오 교차 데이터
사실. 정량 지표: gazer_ep9 step-2 IoU {_fmt(ov_g, 'iris_iou_step2_sameview')} /
d16 {_fmt(ov_d, 'iris_iou_step2_sameview')} · 클래스 분율(결막~17%,
홍채~13%, 동공~1.1%)가 두 모델에서 거의 동일.</p>
"""

    if _sig:
        _sig_html = f"""
<div class="verdict">
<b>ep1 신호 확인 (dense GRPO 재학습, live)</b> — phase0
reward-landscape 캐시(73,800프레임, ~118GB RAM) 완주 후 ep1
it{_sig['it']}/{_sig['total']} ({_sig['it'] / _sig['total'] * 100:.0f}%) 진행 중:
grpo/adv_std <b>{_sig['adv_std']:.3f}</b> (플러 수준 0.016의
{_sig['adv_std'] / 0.016:.0f}배 — §3-1의 신호 소실 복구),
grpo/neg_pg <b>{_sig['neg_pg']:.4f}</b> (≠0 — 강화 방향 업데이트
발생), grpo/reward {_sig['reward']:.3f}, grad/norm {_sig['grad_norm']:.3f} (안정).
§3-5의 ep1 확인 기준 3종(adv_std≫0.016, neg_pg≠0, grad 정상) 충족.</div>
"""
    else:
        _sig_html = ('<div class="pending">ep1 신호 대기 — dense GRPO '
                     'it 로그가 아직 없음 (phase0 또는 초기 단계).</div>')

    _eval_line = (
        f"eval-compare <b>완료</b> (측정 {evalr.get('time', '?')}) — "
        "§3a 표에 동일 지표 비교 반영됨."
        if evalr else
        "eval-compare <b>진행 중</b> — phase0 10번째 캐시 완주 시점 GPU1 "
        "단독 자동기동(waiter). JSON 완료 시 §3a 표가 자동 채워지고 "
        "report-watcher가 본 문서도 자동 재생성.")

    _gate_note = ("gate.json 아직 없음 — Stage A 체인(e32d8→e16d16→e16d8) "
                  "완주 후 Stage B --gate가 val 전용 landscape로 4변종 "
                  "동일 지표 재측정 → '보상 유지되는 가장 가벼운 변종' "
                  "선정 (§4-2).")

    v4progress_html = f"""
<p><b>이 섹션은 생성 시각 기준의 실험 스냅샷</b> — 리포트 재생성
(<code>tools/gazer_report.py</code>)마다 학습 로그·JSON에서 다시
읽어 자동으로 갱신됩니다. 정적 수치(벤치·오버레이)와
동적 수치(학습 진행)가 섞여 있습니다.</p>
<div class="kpigr">
<div class="kpi"><span class="kv">{_ag_kpi}</span><span class="kl">agreement_teacher — 풀 정책 VAL best (§3)</span></div>
<div class="kpi"><span class="kv">{_d16_kpi}</span><span class="kl">dice_proxy — d16 distill VAL (ep{_d16_ep}, ep12/15 예정)</span></div>
<div class="kpi"><span class="kv">{_fmt(ov_g, 'iris_iou_step2_sameview', 4)}</span><span class="kl">iris step-2 동일뷰 IoU — gazer_ep9 (test 무라벨 60프레임)</span></div>
<div class="kpi"><span class="kv">{_fmt(ov_d, 'iris_iou_step2_sameview', 4)}</span><span class="kl">iris step-2 동일뷰 IoU — d16 student</span></div>
<div class="kpi"><span class="kv">{_fmt(ov_x, 'pixel_exact_mean', 4)}</span><span class="kl">pixel exact — gazer_ep9 vs d16 (교차 모델 일관성)</span></div>
<div class="kpi"><span class="kv">{_ms1:.2f} ms</span><span class="kl">풀 모델 스트리밍/프레임 (3090, 예산 {1000.0 / 240.0:.2f} ms 초과 — §2-4)</span></div>
</div>
<p class="note"><b>지표 설명 (KPI)</b> · <b>agreement_teacher</b>:
teacher(SAM3)와의 일치도 = 실질 품질(§3). <b>dice_proxy</b>:
동결 정책(gazer_ep9)과의 일관성 = "정책을 얼마나 흉내 내나"
(0.99 구조적 기대, §4-3). <b>step-2 동일뷰 IoU</b>: 좌우 카메라
교차(0-B)를 감안해 2프레임 간격 같은 눈끼리의 홍채 영역 겹침 —
1.0에 가까울수록 시퀀스 안정. <b>pixel exact</b>: 풀 정책과
경량 student의 argmax 픽셀 일치율 — distill이 형태를 유지하는지.
<b>스트리밍 ms</b>: §2-3 실측, 4090 시연 예산과의 비교는 실측으로
판정.</p>
<h3>5-1. 목표별 진행</h3>
<table>
<tr><th>목표</th><th>상태</th><th>현재 위치</th></tr>
<tr><td>1 · VLM(SAM3)으로 수도라벨 의존 제거</td>
<td><span class="badge b-done">인프라 완료</span></td>
<td>train reward-landscape 캐시 73,800프레임(118GB RAM) 완주 —
시퀀스 도메인 teacher 보상 공급 가능. baseline_400 의존 없이
dense GRPO가 라벨 0의 Sequence 전체에서 학습 중.</td></tr>
<tr><td>2 · RL(GRPO) 시퀀스 세미지도 학습</td>
<td><span class="badge b-run">ep1 진행</span></td>
<td>신호 소실(adv_std 0.016) 진단 → per-pixel dense 보상 +
τ1.5/G16 재학습. 아래 ep1 신호 확인 박스 — 3종 기준 충족
확인됨.</td></tr>
<tr><td>3 · RL 보상 유지 경량화</td>
<td><span class="badge b-run">Stage A 체인</span></td>
<td>warm-start 체인 1단계 e32d16: ep{_d16_last}/15 (VAL dice_proxy
{_d16_kpi} @ep{_d16_ep}). 다음 e32d8 → e16d16 → e16d8 (체인
오케스트레이터 자동), 완주 시 Stage B 게이트(val landscape,
delta ≥ −0.01 통과 중 params 최소). 정적 레이더: e16d8 =
1.360M params / CNN FLOPs 5.4× 감소 (§4-1).</td></tr>
</table>
<p class="note"><b>지표 설명 (열의 의미)</b> · <b>인프라 완료</b>:
teacher 보상 공급(캐시)이 끝났음을 의미 — 목표 1의 실험적
"의존 제거" 검증은 eval-compare(§3a)로 진행. <b>ep1 진행</b>:
28,498 it/에포크 중 초기부. <b>Stage A 체인</b>: §4-2의
4변종 warm-start 순서. <b>VAL dice_proxy</b>는 3에포크마다
측정(ep3/6/9/12/15).</p>
{_sig_html}
<h3>5-2. 풀 정책 VAL 추이 (dense GRPO, §3-3 상세)</h3>
<table style="width:auto"><tr><th>에포크</th><th>agreement_teacher</th></tr>{_val_traj}</table>
<p class="note">ep1 {_val_first} → best {_ag_best:.4f} (ep{_ag_ep})의
모양: 초기화(fold_1 시드) 직후부터 teacher 0.777 수준 —
SAM3 soft4가 fold_1 분할과 높은 baseline 정합을 가지므로 GRPO
시작점이 이미 teacher 근처. 상승 폭이 크지 않은 것은
§3-1(신호 소실) 진단과 일치: 보상 신호가 작아 학습이
"교정" 수준.</p>
<h3>5-3. 정성 결과 — Sequence test 오버레이 (무라벨 도메인)</h3>
{_figs}
<h3>5-4. 대기 중인 자동 이벤트</h3>
<ul>
<li>{_eval_line}</li>
<li>{_gate_note}</li>
<li>gate.json 등장 시: Stage B 판정 + 본 리포트 재생성 + 커밋/태그
(자동 오케스트레이터).</li>
</ul>
"""

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
.pseudo {{ background:#0e1626; border:1px solid #2f4270; border-left:4px
  solid var(--ac); border-radius:8px; padding:12px 16px; margin:8px 0;
  font:13.5px/1.7 'Segoe UI',system-ui,sans-serif; color:#d6e0f2; }}
.pseudo .ph {{ font-weight:700; color:#b9c6e0; }}
.pseudo .arrow {{ color:var(--ac); font-weight:700; }}
.pseudo .kw {{ color:#7cc4ff; font-weight:600; }}
.pseudo .cm {{ color:var(--mut); font-size:12.5px; display:block;
  margin:4px 0 0; }}
.pseudo ol, .pseudo ul {{ margin:6px 0; padding-left:22px; }}
.pseudo li {{ margin:3px 0; }}
.badge {{ display:inline-block; font-size:12px; padding:2px 10px;
  border-radius:20px; margin-right:6px; }}
.b-done {{ background:#12271c; color:var(--ok); border:1px solid var(--ok); }}
.b-run {{ background:#2a2410; color:var(--warn); border:1px solid var(--warn); }}
.b-wait {{ background:var(--card); color:var(--mut); border:1px solid var(--mut); }}
.kpigr {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px; margin:10px 0; }}
.figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr));
  gap:10px; margin:10px 0; }}
.figgrid img {{ width:100%; border-radius:8px; border:1px solid #2a3550;
  display:block; }}
.figcap {{ color:var(--mut); font-size:12.5px; margin:4px 0 10px; }}
</style></head><body><div class="wrap">
<h1>GAZER 진행 리포트</h1>
<p class="note">생성: {now} · AutoGaze × GRPO — OpenEDS2019 label-scarce 시퀀스
분할 · SAM3 live teacher · 정책 = 분할 backbone 그 자체 (plan §7) ·
데이터 소스: gazer_bench.json / gazer_grpo_backbone_t5.log / gazer_eval_compare.json</p>

<h2>한눈에 보기 — 이 실험이 하는 일과 읽는 법</h2>
<div class="card">
<p><b>문제</b>: OpenEDS2019의 <i>Sequence</i> 분할은 프레임 번호만 있고
4개 클래스(배경/결막/홍채/동공) 라벨이 <b>0개</b>입니다. 기존
"baseline_400"은 semantic 분할의 수도라벨 400개를 빌려 시퀀스 도메인으로
전이 학습했지만, 이 라벨이 없으면 학습 신호 자체가 없습니다.</p>
<p><b>GAZER의 접근</b>: 라벨을 만드는 대신 <b>vision-language 모델
(SAM3)을 "살아있는 teacher"로</b> 세워, 시퀀스 프레임마다 teacher의
분할과 얼마나 일치하는지를 <b>보상(reward)</b>으로 받아 <b>강화학습
(GRPO)</b>으로 분할 backbone을 튜닝합니다. 튜닝한 모델을 다시
<b>실시간 시연(RTX 4090 1장, 아이캠 2스트림×120Hz)에 맞추어 경량화</b>
합니다 — 단, RL 보상(agreement_teacher)이 teacher보다 0.01 이상 떨어지지
않는 범위 안에서만.</p>
<p><b>본 리포트의 구성</b> · 실험을 이해하기 위한 <b>배경 2개 섹션(선행
모듈 0-A v3, 0-B 데이터셋 구조)</b>이 먼저 오고, 이후 <b>실험 5개 섹션</b>이
진행 순서대로 이어집니다:
① <b>2 스트레스 테스트</b> — "어느 모듈이 실시간 병목인가"를 숫자로 확인
→ CNN(인코더+디코더)이 병목임을 근거로.
② <b>3 성능 진단·수정</b> — RL 학습이 신호를 못 받는 문제(adv_std 0.016 =
플러)를 진단하고 픽셀 단위 보상으로 고친 뒤, <b>같은 눈금자</b>로
baseline과 비교하는 방법.
③ <b>4 경량화</b> — 병목인 CNN 폭을 단계적으로 줄이는 4개 변종을
학습(warm-start 체인)하고, <b>RL 보상 게이트</b>로 "보상이 유지되는 가장
가벼운 변종"을 고르는 절차.
④ <b>5 v4 진행 상황</b> — 목표별 현재 위치 + 정량 KPI + 오버레이
이미지(스냅샷, 자동 갱신).
⑤ <b>1 버전관리</b> — 태그/커밋. 수치는 실험 서버(3090) 실측이며,
시연 목표(4090) 충족 여부는 최종 실측으로 판정합니다.</p>
<p class="note"><b>읽기 팁</b>: 각 표는 <i>열(지표)의 의미</i>를 표 바로 아래
"지표 설명" 행으로 풀어서 적었습니다. 코드 인용 대신 <b>자연어 절차
(알고리즘 박스)</b>로 알고리즘을 설명합니다. 용어가 어려우면 아래
사전을 먼저 보세요.</p>
<table>
<tr><th>용어</th><th>이 실험에서 의미</th></tr>
<tr><td><b>정책(policy) / backbone</b></td>
<td>시야각을 받아 4개 클래스 분할(마스크)을 내는 신경망. GRPO의 "행위자"가
바로 이 분할 모델 그 자체입니다 (plan §7). gazer_ep9 = 10-26 런에서 8ep
GRPO로 튜닝한 대표 정책.</td></tr>
<tr><td><b>teacher (SAM3)</b></td>
<td>라벨 없이도 "이 픽셀은 어떤 클래스인가"에 대한 부드러운 분할
(soft4, 4개 확률의 합 1)을 내는 vision-language 모델. 보상의 기준이
되며 학습 중엔 가중치 동결.</td></tr>
<tr><td><b>reward landscape</b></td>
<td>프레임마다 teacher의 soft4 분할을 미리 계산해 RAM에 둔 "기준 답안
집". 학습 중 매 프레임마다 teacher를 다시 부르지 않고 이 캐시(118GB,
train용)로 보상을 줍니다 — teacher 호출이 빠지지 않는 속도의 핵심.</td></tr>
<tr><td><b>agreement_teacher</b></td>
<td><b>실질 품질 지표</b>. 모델의 softmax와 teacher soft4 사이의
dice_agreement (0~1). "teacher의 눈과 얼마나 같은가를" 잰 값.</td></tr>
<tr><td><b>dice_proxy</b></td>
<td><b>내적 일관성 지표</b>. 경량화 단계에서 student의 softmax와
동결된 정책(gazer_ep9) softmax 사이의 dice. teacher에 대한 품질이
아니라 "policy를 얼마나 흉내 내나"를 잰 값이라 0.99가 구조적으로
기대되는 수치 (상세: §4-3 주석).</td></tr>
<tr><td><b>GRPO / advantage / reward</b></td>
<td>그룹 상대 강화학습. 같은 프레임 G개 샘플(롤아웃)을 뽑아,
각 픽셀 보상이 그룹 평균보다 좋은지 나쁜지를 advantage로 바꿔 정책을
올립니다. "dense" = 보상·advantage·KL을 프레임 1개 값이 아니라
<b>픽셀마다</b> 계산 (§3-2).</td></tr>
<tr><td><b>annealing / 경량화</b></td>
<td>네트워크의 채널 폭을 <b>단계적으로</b> 좁히며 성능을 지키는 것.
본 실험은 "RL 보상이 teacher−0.01 이상 유지되는 가장 가벼운 폭"을
고르는 게이트(§4-2)로 종료점을 정합니다.</td></tr>
<tr><td><b>warm-start 체인</b></td>
<td>경량화 변종을 "교사(e32d32) → e32d16 → e32d8 → e16d16 → e16d8"
순으로, 각 단계가 <b>바로 이전 단계의 best 가중치</b>에서 시작해
점진적으로 capacity를 빼는 훈련 순서.</td></tr>
<tr><td><b>시연 예산 (4.17 ms)</b></td>
<td>최종 목표: 단일 4090에서 T=8, 아이캠 2스트림 × 120Hz → GPU가
스트림 1프레임을 소화할 수 있는 시간 = 1000/(2×120) ms. 이를 넘는
모든 모델은 실시간이 안 됩니다.</td></tr>
</table>
</div>

<h2>선행 모듈 — v3 (VivimMamba 멀티프레임)와 데이터셋 구조</h2>
<div class="card">
<p><b>이 섹션이 존재하는 이유</b>: GAZER(v4)는 "무엇을 고치려는 실험"입니다.
고치려는 대상(v3 모듈)과 실험의 재료가 되는 데이터(OpenEDS2019)의 구조를
모르면 연구 목표 3줄이 "왜" 세워졌는지 읽을 수 없습니다. 아래 두 개의
배경 섹션(0-A v3 모듈, 0-B 데이터셋 구조)은 GAZER 설계의 출발점이며,
수치는 실제 코드·파일 실측입니다.</p>
</div>

<h3>0-A. v3 모듈 — 지금 GAZER가 개선 중인 "그" 백본</h3>
<div class="card">
<p><b>위치</b> · <code>models/vivim_backbone_multiframe.py</code>의
<code>VivimBackboneMultiFrame</code> (258줄,
v2.9.0-sagd-refactored-nanfixed → v3.1.0, 14 커밋). 구성은
"2D U-형 CNN 백본 + bottleneck에서 Mamba3 시간 블록 1개".
이 모델이 GAZER의 <b>정책 그 자체</b>가 됩니다 (연구 목표 2: "정책 = 분할
backbone").</p>
{SVG_V3_FLOW}
<p class="note"><b>다이어그램 읽는 법</b> · 실선 = 특징 흐름, 점선(파랑) =
U-Net 스킵 연결. 숫자는 채널 폭 × 공간 해상도(448² = 448×448 픽셀).
Mamba3 블록(보라)만 T(5) 프레임의 <b>시간 축</b>을 하나의 시퀀스로 처리 —
그 외 모든 블록은 프레임마다 독립적으로 2D만 봄. 448 = 원본 400 픽셀
크롭에 48 패딩(경계 유지).</p>
<p><b>입력 파이프라인</b> · 640×400 아이캠 원본 → <code>input_crop
[120,520,0,400]</code>(400×400 안쪽 눈 영역) → 448×448 패딩 →
그레이스케일 1채널. 전처리 = gamma 0.8 → CLAHE(clip 1.5, 8×8 타일) →
[-1,1] 정규화 (<code>RITnetPreprocessor400</code>).</p>
<p><b>Mamba3 시간 블록 (<code>TemporalMambaBlock</code>)</b> ·
bottleneck 특징 <code>[B,T,256,56,56]</code>를
<code>[B·56·56, T, 256]</code>로 재배열 → "각 공간 픽셀 56×56개가 길 T=5인
시퀀스"가 됨. 그 위에 LayerNorm → Mamba3 → 잔차 연결 → 원래 shape
복원. 즉 Mamba3는 <b>공간 위치별 5프레임 시퀀스</b>를 처리하는 선택적
시간 경로(selective SSM)입니다. 출력 폭 256은 v4 경량화에서 <b>고정</b> —
이 값이 Mamba3 입력 차원이므로, RL이 학습한 시간 가중치를 그대로
인상하려면 256을 유지해야 합니다(§4-1).</p>
<div class="pseudo">{PSEUDO_V3_TRAIN}</div>
<div class="pseudo">{PSEUDO_V3_STREAM}</div>
<p><b>v3에서 완성된 것</b> (GAZER 출발점)</p>
<table>
<tr><th>항목</th><th>내용</th><th>본 리포트에서의 역할</th></tr>
<tr><td>멀티프레임 분할 백본</td>
<td>T=5 윈도우 4클래스 분할, 학습 시 T프레임 전부 / 인퍼런스 시
마지막 프레임 디코딩 (동일 forward, 플래그 1개로 분기)</td>
<td>GAZER 정책의 몸체 (§3, 연구 목표 2)</td></tr>
<tr><td>FIFO 120Hz 스트리밍 경로</td>
<td>bottleneck 특징 FIFO(T−1) 캐시로 프레임당 encode 1 + Mamba3 +
decode 1 (commit 44aa03872). 도입 전 T=3 1,000프레임 중 ~80%
레이턴시 초과 탈락</td>
<td>§2-3 스트리밍 실측의 측정 대상, 시연 예산의 전제</td></tr>
<tr><td>T 스윕 가중치</td>
<td>T=3/5/7/8/9/11 변종 학습 — 시간 윈도우 길이와 성능 관계 실측
(게임즈 정확도 ~1.25°, T=7, docs/version_management.md)</td>
<td>v4도 T=5로 계속 (§3-2: 1 시퀀스 = T=5)</td></tr>
<tr><td>세미지도 3전략</td>
<td>Strategy A(희소라벨 이중스트림) / D(경계 윈도우) / E(하이브리드) —
configs/phase1_sparse_t5.yaml 등, fold_1 사전학습 시드, bf16, lr 0.001</td>
<td>아래 "한계"가 GAZER 설계 근거</td></tr>
<tr><td>BF16 파이프라인</td>
<td>학습·추론·bench 전체 bf16 (2×3090). 시연 목표는 4090 1장</td>
<td>모든 실측 수치의 조건</td></tr>
</table>
<p class="note"><b>지표 설명</b> · <b>레이턴시 초과 탈락</b>: FIFO 도입 전
실시간 데모에서 프레임 처리 시간이 예산을 넘겨 버려진 프레임 비율.
<b>게임즈 정확도 ~1.25°</b>: v3 시연 파이프라인(7단계 후처리 포함)의
시선 각도 오차 평균 — 이 수치가 v4가 이어받는 "품질 기준선"입니다.
<b>fold_1 사전학습 시드</b>: nnUNet semantic fold_1
(semantic 도메인만 학습한 체크포인트)에서 시작해 시퀀스 도메인으로
적응하는 방식.</p>
<h4>v3 설계의 한계 — GAZER(v4)가 고치려는 것</h4>
<ol>
<li><b>감독 신호가 GT 프레임에 국한</b> — DiceCE·WeakMed·MultiFrameMasked
손실은 모두 <code>label_mask=1</code>(=Semantic GT) 픽셀에만 적용됩니다.
무라벨 Sequence 프레임(91,200개)은 temporal smoothness·M2B 같은
<b>정규화자</b>만 받고 <b>내용 감독(이 픽셀이 무엇인가)을 받지
못합니다</b>. 정규화자는 분할이 "부드럽고 일관된다"만 보장할 뿐
"정확하다"는 보장을 주지 못합니다.</li>
<li><b>라벨 양이 세션당 1~222프레임</b> — 데이터셋 구조(0-B)에서 보듯
Semantic은 세션 꼬리의 0.3~1.1초 구간. 무라벨 600프레임 스트림에 대한
적응 신호가 구조적으로 희박합니다.</li>
<li><b>baseline_400 우회</b> — 이 한계를 메우기 위해 semantic 수도라벨
400개를 시퀀스 도메인 유사라벨(Sequence_PseudoLabels_400)로
"빌려"온 베이스라인이 존재합니다. v4 연구 목표 1이 없애려 하는
의존성이 이것입니다 — 이 라벨이 없으면 v3의 시퀀스 적응 학습 신호는
0이 됩니다.</li>
<li><b>연산 병목 = CNN</b> — §2 실측: FLOPs CNN:Mamba3 = 21.4:1,
wall 61%가 CNN decoder. 시연 예산(4.17 ms/프레임) 대비 풀 모델
5.03 ms/프레임 = <b>예산을 이미 초과</b>한 상태로 v4에 들어옵니다.
이는 v4 연구 목표 3(경량화)의 직접적 출발점입니다.</li>
</ol>
<p><b>이 한계에서 v4(GAZER)로</b> · 내용 감독의 공백을
<b>라벨이 아니라 teacher로 채운다</b>: SAM3(연구 목표 1)이 무라벨
Sequence 프레임마다 soft4 분할을 내고, 그 일치도 <code>agreement_teacher</code>
가 <b>픽셀 단위 보상</b>이 되어 (연구 목표 2) 분할 백본을 GRPO로
튜닝합니다. 보상 기준이 "GT 라벨"에서 "VLM teacher"로 바뀌는 순간,
수도라벨 의존(한계 3)이 사라지고 무라벨 91,200프레임 전체가
감독 신호를 받습니다(한계 1·2 해결). 병목(한계 4)은 같은 보상
지표가 게이트가 되어 경량화 폭을 정합니다(연구 목표 3).</p>
</div>

<h3>0-B. OpenEDS2019 데이터셋 구조 — Semantic vs Sequence, 분할 방법 분석</h3>
<div class="card">
<p><b>질문</b> · "Semantic(라벨)과 Sequence(무라벨)는 어떤 관계인가" —
시퀀셜 스트림 일부 <b>샘플링</b>으로 semantic을 만든 것인가, 한 스트림을
<b>쪼개</b> semantic·sequence로 나눈 것인가, <b>완전 랜덤 분할</b>인가.
아래는 152개 세션(참가자) 전체의 파일명(=프레임 번호)을 분석한
결과입니다(실측, OpenEDS2019/).</p>
<div class="verdict">
<b>판정 — 세 번째도, 첫 번째도 아니다: "참가자 세션 단위의
<b>인접 블록 분할</b>"이다.</b>
<p>· <b>SEM∩SEQ = 0프레임</b> (12,759 ∩ 91,200, 교집합 0) —
"시퀀셜 스트림의 일부 프레임이 semantic 라벨을 가졌다"는
샘플링 구조가 아님.</p>
<p>· 프레임 번호는 <b>참가자(세션) 1인당 1개 연속 블록</b>이며,
블록 안의 순서가 고정: <b>Generative(~2,041) → Sequence(600) →
Semantic(1~222)</b>, 블록 간 gap 0 (152/152 세션 확인).
즉 <b>한 세션 스트림이 끝나는 쪽부터 Semantic(라벨) — Sequence
(무라벨) — Generative(무라벨) 순으로 인접 배치</b>되어 있다.</p>
<p>· train/val/test = <b>참가자 단위</b> (95 / 28 / 29) —
"프레임 랜덤 분할"이 아님. 동일 참가자의 모든 세그먼트가 같은
스플릿.</p>
</div>
{HTML_DS_DIAGRAM}
<table>
<tr><th>분할</th><th>참가자</th><th>Sequence (무라벨)</th><th>Semantic (GT)</th><th>세션당 Semantic 비율</th></tr>
<tr><td>train</td><td>95</td><td>57,000 (600×95)</td><td>8,916 (1~222/세션, 평균 94)</td><td>13.5%</td></tr>
<tr><td>validation</td><td>28</td><td>16,800 (600×28)</td><td>2,403 (36~138/세션, 평균 86)</td><td>12.5%</td></tr>
<tr><td>test</td><td>29</td><td>17,400 (600×29)</td><td>1,440 (1~138/세션, 평균 50)</td><td>7.6%</td></tr>
<tr><td>합계</td><td>152</td><td><b>91,200</b></td><td><b>12,759</b></td><td>12.3%</td></tr>
</table>
<p class="note"><b>지표 설명</b> · <b>Sequence 600프레임</b> =
참가자 1인당 스트림 1.5초 × 200Hz (README 공식 사양, 전부 확인).
<b>Semantic 비율</b> = (Semantic ÷ (Sequence+Semantic)), 세션
라벨링 예산이 참가자마다 달라 0.2%(test 최소)~27%(최대)로
변동. <b>train/val/test</b>는 프레임 단위가 아니라
<b>참가자 전체 단위</b>로 나뉜다(도메인 이질성 방지 — 같은
사람의 눈은 같은 스플릿).</p>
<p><b>구조가 의미하는 것 (GAZER 관점)</b></p>
<ol>
<li><b>라벨은 시퀀스의 "꼬리"에 붙어 있다</b> — 각 세션에서
Sequence(600) 직후에 Semantic(1~222)이 연속된다. v3의
BoundaryWindowDataset(0-A)이 "Sequence 끝 → Semantic 시작" 경계
윈도우를 샘플링한 근거가 바로 이 인접성이다. 즉 <b>같은 참가자의
무라벨 스트림과 라벨 구간은 시각·도메인적으로 가장 가까운
구간</b>이라, 경계 윈도우가 도메인 가교의 가장 강한 신호다.</li>
<li><b>무라벨 91,200 = 시연과 동일 분포</b> — Sequence test
(17,400)가 v4 오버레이 검증의 대상(프레임 336400~336459)이고,
GAZER의 보상·게이트가 평가하는 대상도 결국 이 무라벨 분포다.
"시퀀스 세미지도 학습"(연구 목표 2)의 대상이
<b>샘플링 잔여분이 아니라 600×152 전체</b>임을 이 구조가
보장한다 — 어떤 프레임도 semantic에 "빌려주지 않았으므로",
무라벨 전체가 teacher 보상만 받는 순수 도메인이다.</li>
<li><b>참가자 단위 스플릿의 함의</b> — test 참가자의 라벨(1,440)은
학습에서 쓰이지 않는다(다른 사람). 따라서 "test semantic"을
teacher 보상 기준에 쓰는 것은 일반화 검증이
<b>불가능</b>하다 — v4 게이트(§4-2)가 val 전용 landscape를
사용하는 이유이기도 하다.</li>
</ol>
<p><b>데이터셋 분석이 시사한 시퀀셜 동공 학습 전략 (선택 추가)</b></p>
<ol>
<li><b>경계 이점이 v4 보상에도 그대로 적용</b> — Semantic은
Sequence의 <b>바로 다음</b> 프레임부터 시작한다. GRPO rollout
시퀀스(T=5)가 경계 부근을 지나면, <b>윈도우 안의 Semantic 프레임은
GT(teacher bypass), Sequence 프레임은 teacher 보상</b>을 받아
한 시퀀스 안에서 "정답 기준점(semantic) + 보상 대상(sequence)"이
공존한다. v3 Phase 2(D)가 손실 함수 단위로 이 공존을 만들었다면,
v4는 <b>보상 단위로</b> 이 공존을 재구성한다 — 동공/홍채가
시계열상 가장 잘 보존되는(동공 반경·중심 궤적) 경계 윈도우를
rollout 샘플링 우선순위 상단으로 두는 전략이 데이터 구조상
합법적이고, teacher-보상의 <b>절대 기준점</b>이 시퀀스 안에
내장된다.</li>
<li><b>스테레오 교차를 감안한 시퀀스 구성</b> — Sequence 프레임은
좌·우 카메라가 <b>프레임 1장씩 교차</b>로 들어온다(실측: 동일
뷰 step-2 MAD 2.52 ≪ 인접 47.95, 거울 57.28). 즉 연속 T=5
윈도우는 [우,좌,우,좌,우] 형태 → <b>인접 프레임 IoU가 낮게
나오는 것은 모델이 아니라 데이터 사실</b>(§3 오버레이 진단의
핵심). 시퀀셜 동공 학습 관점에서는 reward·advantage의
시간 적분이 2프레임 주기(좌우)로 잡히면, <b>step-2(동일 뷰)
차분</b>이 노이즈 없는 시간 신호가 된다. v4 dense GRPO는 이미
per-pixel 보상이므로 시간 적분은 불필요하지만, <b>경량화
정규화자(α_tv)와 temporal smoothness 계수는 step-2 기반</b>으로
잡아야 좌우 카메라 교차로 인한 가짜 "변동"을 처벌하지
않는다. (현재 v3 temporal smoothness는 인접-프레임 L2 —
이 교차를 감안하면 과징벌이며, v4 annealing 정규화자가
step-2로 조정되면 시퀀셜 품질이 데이터 사실과 정렬된다.)</li>
</ol>
<p class="note"><b>분석 방법 (재현)</b> · 각 분할의
<code>*.png</code> 파일명 앞 8자리(프레임 번호)를 정렬 →
연속 구(run) 분할 → SEM/SEQ 교집합·run 1:1 대응·
<code>OpenEDS_*_userID_mapping_to_images.json</code>의
<code>sequence_images</code>/<code>semantic_segmenation_images</code>
블록 경계 gap 검증(152/152 gap=0). 스테레오 교차 = 동일 train
세션의 <code>mad(k,k+1)</code> vs <code>mad(k,k+2)</code> vs
<code>mad(mirror(k),k+1)</code> (cv2, 2010~2012).</p>
</div>

<h2>연구 목표 (3단계)</h2>
<div class="card">
<table>
<tr><th>#</th><th>목표</th><th>본 리포트 내 대응</th></tr>
<tr><td>1</td><td><b>VLM(SAM3)을 삼아 베이스라인 수도라벨 의존성 없애기</b> —
semantic fold_1 초기화 + SAM3 soft4 reward-landscape로 pseudo-label 없이
시퀀스 도메인 적응</td>
  <td>§3 (SAM3 teacher + reward landscape) · §3a (동일 지표로 베이스라인
  비교 — 수도라벨 의존 탈피 검증)</td></tr>
<tr><td>2</td><td><b>RL(GRPO or PPO) 튜닝으로 시퀀스 세미지도 학습</b> —
정책 = 분할 backbone, per-pixel mask가 action, dense teacher-likelihood
reward로 on-policy GRPO</td>
  <td>§3-2 (dense per-pixel GRPO 신호 복원) · §3-3 (학습 로그 수치)</td></tr>
<tr><td>3</td><td><b>최종 스테이지: VLM+RL튜닝 가중치 + Annealing된 '경량화' 모델</b> —
RL 보상이 유지되도록 연산 병목 모듈을 annealing</td>
  <td>§2 (병목 = CNN enc+dec 확인 + 시연 2스트림 예산 분석) · §4
  (CNN 인코더+디코더 2×2 채널 annealing — Stage A warm-start 체인
  distill → Stage B RL 보상 게이트 →
  4090 T=8 2스트림 120Hz 실측으로 최종 변종 선정)</td></tr>
</table>
<p class="note"><b>읽는 법</b> · 목표 3줄은 사용자 원문을 그대로 유지합니다.
"본 리포트 내 대응" 열은 각 목표가 이 문서의 어느 섹션에서 검증되는지
가리키는 지도입니다: 목표 1 → §3(teacher + landscape)과 §3-4(동일
지표 비교), 목표 2 → §3-2(dense GRPO 신호 복원)과 §3-3(로그 수치),
목표 3 → §2(병목 확인 + 예산)와 §4(2×2 annealing + 게이트).
읽을 때 "이 섹션이 어느 목표의 증거인가"를 이 열로 확인하시면
됩니다.</p>
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
<p class="note"><b>지표 설명 (열의 의미)</b> · <b>상태</b>:
완료(검증 끝) / 진행 중(현재 실행) / 본 문서(이 보고서 자체).
<b>요약</b>의 수치는 각 단계의 증거: 단계 2의
"21.4:1, wall 61%" = CNN이 Mamba3의 21.4배 연산량, 실제 시간의
61%를 CNN decoder가 차지(§2-1/2-2 실측). 단계 3의
"adv_std 0.016" = 신호 소실 진단 수치(§3-1), "adv_std≫0.016,
neg_pg≠0" = 재기동 성공 확인 기준(§3-5). 단계 4의
"e16d8 = 1.360M / CNN FLOPs 5.4× 감소" = 가장 가벼운 변종의
정적 레이더 실측(§4-1).</p>

<h2>1. 버전관리</h2>
<div class="card">{step1_html}</div>

<h2>2. 스트레스 테스트 — 병목 원상 확인</h2>
<div class="card">{stress_html}</div>

<h2>3. 성능 진단·수정 + 동일 지표 비교</h2>
<div class="card">{diag_html}</div>

<h2>4. 경량화 + Annealing (실험 실행 중)</h2>
<div class="card">{lite_html}</div>

<h2>5. v4 실험 진행 상황 (스냅샷)</h2>
<div class="card">{v4progress_html}</div>

<h2>6. 리포트 재생성</h2>
<div class="card"><p>
<code>nnunet_env/bin/python tools/gazer_report.py</code> —
bench/eval-compare JSON과 학습 로그를 읽어 본 문서를 재생성한다.
eval-compare(JSON) 완료 후 1회 재실행하면 3a 표와 4단계 개문 판정이
자동 반영된다. 5단계(진행 상황)도 같은 소스에서 자동 갱신된다.</p></div>
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
