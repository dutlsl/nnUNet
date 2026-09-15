#!/usr/bin/env python3
import os, re, sys, time, subprocess

PROJECT_ROOT = '/home/iulab1/PycharmProjects/nnUNet'
LOG1 = os.path.join(PROJECT_ROOT, 'logs/phase1_sparse_t5.log')
LOG2 = os.path.join(PROJECT_ROOT, 'logs/phase2_multiframe_t5.log')

def parse(path):
    if not os.path.exists(path):
        return {'ep': '-', 'loss': '-', 'val': '-', 'best': '-', 'prog': 'Waiting...'}
    try:
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(path) - 80000))
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
    except Exception:
        return {'ep': '-', 'loss': '-', 'val': '-', 'best': '-', 'prog': 'Error'}
    
    eps = [l.strip() for l in lines if re.match(r'^E\d{4}/\d{4}', l.strip())]
    bests = [l.strip() for l in lines if '★ Best Val Dice' in l]
    tqdms = [l.strip() for l in lines if 'Train E' in l]
    
    latest_ep = eps[-1] if eps else '-'
    latest_best = bests[-1] if bests else '-'
    prog = tqdms[-1] if tqdms else 'Running'
    if 'it/s' in prog:
        m = re.search(r'(Train E\d+:\s+\d+%.*?loss=[\d\.]+.*)', prog)
        if m: prog = m.group(1)
    
    return {'ep_line': latest_ep, 'best': latest_best, 'prog': prog}

def get_gpus():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], timeout=2).decode().strip().splitlines()
        return [p.split(',') for p in out]
    except Exception:
        return []

while True:
    p1 = parse(LOG1)
    p2 = parse(LOG2)
    gpus = get_gpus()
    
    buf = []
    buf.append('[2J[H')
    buf.append('[1;37;44m' + ' '*82 + '[0m')
    buf.append('[1;37;44m   📊 EYEBALL-3D TRAINING & SERVER LIVE DASHBOARD (Clean Mode)   ' + ' '*18 + '[0m')
    buf.append('[1;37;44m' + ' '*82 + '[0m
')
    
    buf.append('[1;33m[1. DUAL GPU RESOURCE STATUS][0m')
    for g in gpus:
        if len(g) >= 7:
            idx, name, temp, util, mu, mt, pwr = [x.strip() for x in g[:7]]
            role = 'Phase 1 (GPU 0)' if idx == '0' else 'Phase 2 (GPU 1)'
            buf.append(f'  GPU {idx}: [1;32m{util:>3}%[0m Util | [1;36m{mu:>5}MB[0m/{mt}MB | {temp}°C | {float(pwr):.0f}W  --> [1m{role}[0m')
    buf.append('')
    
    buf.append('[1;36m' + '='*82 + '[0m')
    buf.append('[1;36m[2. PHASE 1: GPU 0 (Loss 분기 + DualStream)][0m')
    buf.append(f'  • Current Progress : {p1["prog"]}')
    buf.append(f'  • Latest Result    : [1m{p1["ep_line"]}[0m')
    buf.append(f'  • Milestone        : [1;33m{p1["best"]}[0m')
    buf.append('')
    
    buf.append('[1;35m' + '='*82 + '[0m')
    buf.append('[1;35m[3. PHASE 2: GPU 1 (Multi-Frame 디코딩 + Masked Loss)][0m')
    buf.append(f'  • Current Progress : {p2["prog"]}')
    buf.append(f'  • Latest Result    : [1m{p2["ep_line"]}[0m')
    buf.append(f'  • Milestone        : [1;33m{p2["best"]}[0m')
    buf.append('[1;35m' + '='*82 + '[0m')
    buf.append('[2m(실시간 2초 갱신 | 종료: Ctrl+C)[0m')
    
    print('
'.join(buf), flush=True)
    time.sleep(2)
