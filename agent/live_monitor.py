#!/usr/bin/env python3
"""
Real-time Clean Terminal Monitor & Hardware Dashboard for 3D Eyeball Research.
Features:
  - Full Hardware Specs: CPU (model/cores/util%/load avg), RAM (used/avail/total/swap), Disks (root & ssd1), GPUs (util%, VRAM, temp, fan, power/cap)
  - Epoch-only Training Metrics for Phase 1 (T=8) & Phase 2 (T=8)
  - Zero terminal clutter: in-place screen refresh, no tqdm line spamming
  - Responsive layout: dynamically adapts to terminal resize without broken line wraps or cut-off columns
"""

import os
import sys
import re
import time
import shutil
import subprocess
import argparse
import psutil

PROJECT_ROOT = "/home/iulab1/PycharmProjects/nnUNet"
LOG_PH1 = os.path.join(PROJECT_ROOT, "logs/phase1_sparse_t8_train.log")
LOG_PH2 = os.path.join(PROJECT_ROOT, "logs/phase2_multiframe_t8_train.log")
PID_PH1 = os.path.join(PROJECT_ROOT, "logs/phase1_sparse_t8.pid")
PID_PH2 = os.path.join(PROJECT_ROOT, "logs/phase2_multiframe_t8.pid")

# ANSI Color Codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[96m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"
C_WHITE = "\033[97m"

def get_cpu_model():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    raw = line.split(":", 1)[1].strip()
                    raw = raw.replace("(R)", "").replace("(TM)", "").replace("CPU", "").strip()
                    return re.sub(r"\s+", " ", raw)
    except Exception:
        pass
    return "Intel Core Processor"

CPU_MODEL_NAME = get_cpu_model()

def make_bar(percent, length=10):
    percent = max(0.0, min(100.0, percent))
    filled = int(round(length * percent / 100.0))
    filled = max(0, min(length, filled))
    bar = "■" * filled + "░" * (length - filled)
    if percent >= 90:
        return f"{C_RED}{bar}{C_RESET}"
    elif percent >= 75:
        return f"{C_YELLOW}{bar}{C_RESET}"
    else:
        return f"{C_GREEN}{bar}{C_RESET}"

def visible_len(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))

def fit_line_ansi(s, width):
    vlen = visible_len(s)
    if vlen == width:
        return s
    elif vlen < width:
        return s + " " * (width - vlen)
    else:
        result = []
        cur_vlen = 0
        i = 0
        n = len(s)
        while i < n:
            if s[i] == "\033" and i + 1 < n and s[i+1] == "[":
                m = re.match(r"^\033\[[0-9;]*m", s[i:])
                if m:
                    result.append(m.group(0))
                    i += len(m.group(0))
                    continue
            if cur_vlen < width:
                result.append(s[i])
                cur_vlen += 1
            i += 1
        result.append(C_RESET)
        return "".join(result)

def get_system_specs():
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_pct = psutil.cpu_percent(interval=None)
    load1, load5, load15 = psutil.getloadavg()
    
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    disk_root = psutil.disk_usage("/")
    disk_ssd = psutil.disk_usage("/mnt/ssd1") if os.path.exists("/mnt/ssd1") else None
    
    gpus = []
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], timeout=2).decode("utf-8").strip().splitlines()
        for line in out:
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 9:
                gpus.append({
                    "idx": p[0],
                    "name": p[1],
                    "temp": float(p[2]),
                    "fan": float(p[3]) if p[3] != "[Not Supported]" else 0.0,
                    "power_draw": float(p[4]),
                    "power_cap": float(p[5]),
                    "util": float(p[6]),
                    "mem_used": float(p[7]),
                    "mem_total": float(p[8]),
                })
    except Exception:
        pass
        
    return {
        "cpu_pct": cpu_pct,
        "cpu_count": cpu_count,
        "cpu_phys": cpu_phys,
        "load": (load1, load5, load15),
        "mem": mem,
        "swap": swap,
        "disk_root": disk_root,
        "disk_ssd": disk_ssd,
        "gpus": gpus,
    }

def get_pid_status(pid_path):
    if not os.path.exists(pid_path):
        return None, "NOT_FOUND"
    try:
        with open(pid_path, "r") as f:
            pid = int(f.read().strip())
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            return pid, p.status()
        return pid, "DEAD"
    except Exception:
        return None, "ERROR"

def parse_training_log(log_path):
    if not os.path.exists(log_path):
        return {"completed": [], "current": "Log file not created yet", "best_dice": 0.0}

    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 400000))
            raw = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return {"completed": [], "current": "Reading error", "best_dice": 0.0}

    lines = raw.splitlines()
    completed = []
    best_so_far = 0.0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_done = re.search(r"\[Epoch\s+(\d+)\s+Done\s+\|\s+([\d\.]+s)\s+\|\s+LR:\s+([\d\.e\-]+)\]", line)
        if m_done:
            ep_num = int(m_done.group(1))
            ep_time = m_done.group(2)
            ep_lr = m_done.group(3)
            
            tr_loss, vl_loss = "-", "-"
            val_dice, pupil, iris, sclera = "-", "-", "-", "-"
            is_new_best = False
            
            for j in range(1, 6):
                if i + j < len(lines):
                    sub = lines[i + j].strip()
                    m_loss = re.search(r"train_loss:\s+([\d\.]+)\s+\|\s+val_loss:\s+([\d\.]+)", sub)
                    if m_loss:
                        tr_loss = float(m_loss.group(1))
                        vl_loss = float(m_loss.group(2))
                    m_dice = re.search(r"Val Dice:\s+([\d\.]+)\s+\(Pupil:([\d\.]+)\s+Iris:([\d\.]+)\s+Sclera:([\d\.]+)\)", sub)
                    if m_dice:
                        val_dice = float(m_dice.group(1))
                        pupil = float(m_dice.group(2))
                        iris = float(m_dice.group(3))
                        sclera = float(m_dice.group(4))
                        if val_dice > best_so_far:
                            best_so_far = val_dice
                            is_new_best = True
                    if "New Best Val Dice" in sub:
                        is_new_best = True

            completed.append({
                "epoch": ep_num,
                "time": ep_time,
                "lr": ep_lr,
                "train_loss": tr_loss,
                "val_loss": vl_loss,
                "val_dice": val_dice,
                "pupil": pupil,
                "iris": iris,
                "sclera": sclera,
                "is_best": is_new_best,
                "best_dice": best_so_far
            })
        i += 1

    current_status = "Standby / Starting..."
    for line in reversed(lines):
        line = line.strip()
        m_tqdm = re.search(r"(\[E\d+\])\s+(\d+%)\|.*?\|\s+(\d+/\d+\s+\[[\d:<]+\])\s*,\s*Loss:([\d\.]+)", line)
        if m_tqdm:
            ep_tag = m_tqdm.group(1)
            pct_tag = m_tqdm.group(2)
            time_tag = m_tqdm.group(3)
            loss_val = m_tqdm.group(4)
            m_scores = re.search(r"Pupil:([\d\.]+)\s+Iris:([\d\.]+)\s+Sclera:([\d\.]+)", line)
            if m_scores:
                current_status = f"{ep_tag} {pct_tag} ({time_tag}) | Loss: {loss_val} | Running: P:{m_scores.group(1)} I:{m_scores.group(2)} S:{m_scores.group(3)}"
            else:
                current_status = f"{ep_tag} {pct_tag} ({time_tag}) | Loss: {loss_val}"
            break
        elif "[Epoch" in line and "Done" in line:
            current_status = "Epoch completed, running validation..."
            break

    return {
        "completed": completed,
        "current": current_status,
        "best_dice": best_so_far
    }

def format_epoch_table(completed_list, width):
    lines = []
    if not completed_list:
        lines.append(fit_line_ansi(f"   {C_DIM}(Epoch 1 in progress... metrics appear here upon epoch completion){C_RESET}", width))
        return lines

    show_recent = completed_list[-5:]
    
    if width >= 106:
        # Full Wide Mode (fits in 99 chars)
        hdr = f"  {'Epoch':^7} | {'Time':^6} | {'LR':^8} | {'Tr.Loss':^7} | {'Val.Loss':^7} | {'MeanDice':^8} | {'Pupil':^6} | {'Iris':^6} | {'Sclera':^6} | {'Status'}"
        lines.append(fit_line_ansi(C_DIM + hdr + C_RESET, width))
        for ep in show_recent:
            star = f"{C_BOLD}{C_YELLOW}★ BEST{C_RESET}" if ep["is_best"] else ""
            m_dice = f"{C_BOLD}{C_GREEN}{ep['val_dice']:.4f}{C_RESET}" if isinstance(ep['val_dice'], float) else str(ep['val_dice'])
            tr_l = f"{ep['train_loss']:.4f}" if isinstance(ep['train_loss'], float) else str(ep['train_loss'])
            vl_l = f"{ep['val_loss']:.4f}" if isinstance(ep['val_loss'], float) else str(ep['val_loss'])
            p_d = f"{ep['pupil']:.4f}" if isinstance(ep['pupil'], float) else str(ep['pupil'])
            i_d = f"{ep['iris']:.4f}" if isinstance(ep['iris'], float) else str(ep['iris'])
            s_d = f"{ep['sclera']:.4f}" if isinstance(ep['sclera'], float) else str(ep['sclera'])
            lr_s = f"{float(ep['lr']):.1e}" if ep['lr'] != "-" else "-"
            
            row = (
                f"  E{ep['epoch']:03d}   | {ep['time']:>6} | {lr_s:>8} | "
                f"{tr_l:>7} | {vl_l:>7} | {m_dice} | {p_d:>6} | {i_d:>6} | {s_d:>6} | {star}"
            )
            lines.append(fit_line_ansi(row, width))
            
    elif width >= 90:
        # Medium Mode (fits in 88 chars)
        hdr = f"  {'Epoch':^6} | {'Time':^6} | {'Tr.Loss':^7} | {'Val.Loss':^7} | {'MeanDice':^8} | {'Pupil':^6} | {'Iris':^6} | {'Sclera':^6} | {'Status'}"
        lines.append(fit_line_ansi(C_DIM + hdr + C_RESET, width))
        for ep in show_recent:
            star = f"{C_BOLD}{C_YELLOW}★BEST{C_RESET}" if ep["is_best"] else ""
            m_dice = f"{C_BOLD}{C_GREEN}{ep['val_dice']:.4f}{C_RESET}" if isinstance(ep['val_dice'], float) else str(ep['val_dice'])
            tr_l = f"{ep['train_loss']:.4f}" if isinstance(ep['train_loss'], float) else str(ep['train_loss'])
            vl_l = f"{ep['val_loss']:.4f}" if isinstance(ep['val_loss'], float) else str(ep['val_loss'])
            p_d = f"{ep['pupil']:.4f}" if isinstance(ep['pupil'], float) else str(ep['pupil'])
            i_d = f"{ep['iris']:.4f}" if isinstance(ep['iris'], float) else str(ep['iris'])
            s_d = f"{ep['sclera']:.4f}" if isinstance(ep['sclera'], float) else str(ep['sclera'])
            
            row = (
                f"  E{ep['epoch']:03d}  | {ep['time']:>6} | "
                f"{tr_l:>7} | {vl_l:>7} | {m_dice} | {p_d:>6} | {i_d:>6} | {s_d:>6} | {star}"
            )
            lines.append(fit_line_ansi(row, width))
            
    else:
        # Compact Mode (fits in 75 chars)
        hdr = f" {'Ep':^5} | {'Tr.Loss':^7} | {'Val.Loss':^7} | {'MeanDice':^8} | {'Pupil':^6} | {'Iris':^6} | {'Sclera':^6} | {'Status'}"
        lines.append(fit_line_ansi(C_DIM + hdr + C_RESET, width))
        for ep in show_recent:
            star = f"{C_BOLD}{C_YELLOW}★BEST{C_RESET}" if ep["is_best"] else ""
            m_dice = f"{C_BOLD}{C_GREEN}{ep['val_dice']:.4f}{C_RESET}" if isinstance(ep['val_dice'], float) else str(ep['val_dice'])
            tr_l = f"{ep['train_loss']:.4f}" if isinstance(ep['train_loss'], float) else str(ep['train_loss'])
            vl_l = f"{ep['val_loss']:.4f}" if isinstance(ep['val_loss'], float) else str(ep['val_loss'])
            p_d = f"{ep['pupil']:.4f}" if isinstance(ep['pupil'], float) else str(ep['pupil'])
            i_d = f"{ep['iris']:.4f}" if isinstance(ep['iris'], float) else str(ep['iris'])
            s_d = f"{ep['sclera']:.4f}" if isinstance(ep['sclera'], float) else str(ep['sclera'])
            
            row = (
                f" E{ep['epoch']:03d} | "
                f"{tr_l:>7} | {vl_l:>7} | {m_dice} | {p_d:>6} | {i_d:>6} | {s_d:>6} | {star}"
            )
            lines.append(fit_line_ansi(row, width))

    return lines

def render_dashboard():
    term_size = shutil.get_terminal_size((105, 36))
    W = max(76, term_size.columns)
    
    specs = get_system_specs()
    ph1 = parse_training_log(LOG_PH1)
    ph2 = parse_training_log(LOG_PH2)
    
    pid1, st1 = get_pid_status(PID_PH1)
    pid2, st2 = get_pid_status(PID_PH2)
    
    out_lines = []
    
    border_top = "╔" + "═" * (W - 2) + "╗"
    border_mid = "╠" + "═" * (W - 2) + "╣"
    border_bot = "╚" + "═" * (W - 2) + "╝"
    
    out_lines.append(C_CYAN + border_top + C_RESET)
    now_str = time.strftime("%H:%M:%S")
    if W >= 90:
        title_str = f" ★ 3D EYEBALL RESEARCH: MULTI-GPU SERVER & TRAINING MONITOR (T=8) ★ "
        hdr_text = f" {C_BOLD}{title_str}{C_RESET}[ {C_YELLOW}{now_str}{C_RESET} ]"
    else:
        title_str = f" ★ 3D EYEBALL RESEARCH: T=8 MONITOR ★ "
        hdr_text = f" {C_BOLD}{title_str}{C_RESET}[ {C_YELLOW}{now_str}{C_RESET} ]"
        
    out_lines.append(fit_line_ansi(f"║{hdr_text}", W - 1) + C_CYAN + "║" + C_RESET)
    out_lines.append(C_CYAN + border_mid + C_RESET)

    bar_len = 8 if W < 95 else 12
    cpu_bar = make_bar(specs["cpu_pct"], bar_len)
    mem_pct = specs["mem"].percent
    mem_bar = make_bar(mem_pct, bar_len)
    swap_pct = specs["swap"].percent
    
    # CPU
    cpu_line = (
        f"  {C_BOLD}CPU{C_RESET}: {specs['cpu_pct']:>5.1f}% [{cpu_bar}] "
        f"| {specs['cpu_phys']}C/{specs['cpu_count']}T ({CPU_MODEL_NAME.split('@')[0].strip()}) "
        f"| Load: {specs['load'][0]:.2f}, {specs['load'][1]:.2f}, {specs['load'][2]:.2f}"
    )
    out_lines.append(fit_line_ansi(cpu_line, W))
    
    # RAM & Swap
    mem_line = (
        f"  {C_BOLD}RAM{C_RESET}: {mem_pct:>5.1f}% [{mem_bar}] "
        f"{specs['mem'].used / 1024**3:5.1f}/{specs['mem'].total / 1024**3:5.1f} GB "
        f"(Free: {specs['mem'].available / 1024**3:5.1f} GB) "
        f"| Swap: {swap_pct:.1f}% ({specs['swap'].used/1024**3:.1f}/{specs['swap'].total/1024**3:.1f} GB)"
    )
    out_lines.append(fit_line_ansi(mem_line, W))
    
    # Disks
    d_root = specs["disk_root"]
    d_ssd = specs["disk_ssd"]
    ssd_str = f"/mnt/ssd1: {d_ssd.used/1024**4:4.2f}/{d_ssd.total/1024**4:4.2f} TB ({d_ssd.percent}%) Free:{d_ssd.free/1024**3:4.0f} GB" if d_ssd else "No /mnt/ssd1"
    disk_line = (
        f"  {C_BOLD}DISK{C_RESET}: /: {d_root.used/1024**3:3.0f}/{d_root.total/1024**3:3.0f} GB ({d_root.percent}%) Free:{d_root.free/1024**3:3.0f} GB | {ssd_str}"
    )
    out_lines.append(fit_line_ansi(disk_line, W))
    
    # GPUs
    for g in specs["gpus"]:
        g_idx = g["idx"]
        g_util_bar = make_bar(g["util"], bar_len)
        vram_pct = (g["mem_used"] / g["mem_total"]) * 100
        vram_bar = make_bar(vram_pct, bar_len)
        pwr_pct = (g["power_draw"] / g["power_cap"]) * 100
        
        job_tag = f"{C_GREEN}Phase 1 (T=8){C_RESET}" if g_idx == "0" else f"{C_BLUE}Phase 2 (T=8){C_RESET}"
        
        if W >= 105:
            gpu_line = (
                f"  {C_BOLD}GPU {g_idx}{C_RESET}: {g['util']:>3.0f}% [{g_util_bar}] "
                f"| VRAM: {g['mem_used']/1024:>4.1f}/{g['mem_total']/1024:>4.1f} GB ({vram_pct:>2.0f}%) "
                f"| Pwr: {g['power_draw']:>5.1f}W/{g['power_cap']:>3.0f}W ({pwr_pct:>3.0f}%) "
                f"| {g['temp']:>2.0f}°C (Fan {g['fan']:>2.0f}%) | {job_tag}"
            )
        else:
            gpu_line = (
                f"  {C_BOLD}GPU {g_idx}{C_RESET}: {g['util']:>3.0f}% [{g_util_bar}] "
                f"| VRAM: {g['mem_used']/1024:>4.1f}/{g['mem_total']/1024:>4.1f}G ({vram_pct:>2.0f}%) "
                f"| Pwr: {g['power_draw']:>5.1f}W ({pwr_pct:>2.0f}%) "
                f"| {g['temp']:>2.0f}°C | {job_tag}"
            )
        out_lines.append(fit_line_ansi(gpu_line, W))
    
    out_lines.append(C_CYAN + border_mid + C_RESET)

    st1_color = C_GREEN if st1 in ("running", "sleeping") else C_RED
    if W >= 100:
        h1 = f" {C_BOLD}{C_GREEN}▶ PHASE 1: Sparse-Label 비지도 (T=8){C_RESET} [GPU 0 | PID: {pid1} ({st1_color}{st1}{C_RESET})] | Best Dice: {C_BOLD}{C_GREEN}{ph1['best_dice']:.4f}{C_RESET}"
    else:
        h1 = f" {C_BOLD}{C_GREEN}▶ PH 1: Sparse (T=8){C_RESET} [GPU 0 | PID: {pid1}] | Best Dice: {C_BOLD}{C_GREEN}{ph1['best_dice']:.4f}{C_RESET}"
    out_lines.append(fit_line_ansi(h1, W))
    
    curr1 = f"   {C_DIM}Live Status:{C_RESET} {ph1['current']}"
    out_lines.append(fit_line_ansi(curr1, W))
    out_lines.extend(format_epoch_table(ph1["completed"], W))

    out_lines.append(C_CYAN + border_mid + C_RESET)

    st2_color = C_GREEN if st2 in ("running", "sleeping") else C_RED
    if W >= 100:
        h2 = f" {C_BOLD}{C_BLUE}▶ PHASE 2: Multi-Frame 전 프레임 디코딩 (T=8){C_RESET} [GPU 1 | PID: {pid2} ({st2_color}{st2}{C_RESET})] | Best Dice: {C_BOLD}{C_BLUE}{ph2['best_dice']:.4f}{C_RESET}"
    else:
        h2 = f" {C_BOLD}{C_BLUE}▶ PH 2: Multi-Frame (T=8){C_RESET} [GPU 1 | PID: {pid2}] | Best Dice: {C_BOLD}{C_BLUE}{ph2['best_dice']:.4f}{C_RESET}"
    out_lines.append(fit_line_ansi(h2, W))
    
    curr2 = f"   {C_DIM}Live Status:{C_RESET} {ph2['current']}"
    out_lines.append(fit_line_ansi(curr2, W))
    out_lines.extend(format_epoch_table(ph2["completed"], W))

    out_lines.append(C_CYAN + border_bot + C_RESET)
    
    frame = "\033[H\033[2J" + "\n".join(out_lines) + "\n"
    sys.stdout.write(frame)
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Live Multi-GPU Training and Server Spec Dashboard")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2.0s)")
    args = parser.parse_args()
    
    if args.once:
        render_dashboard()
        return
        
    try:
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        while True:
            render_dashboard()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
