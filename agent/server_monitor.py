import os, sys, time, psutil, subprocess

def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]).decode('utf-8').strip().split('\n')
        gpus = []
        for line in out:
            parts = [p.strip() for p in line.split(',')]
            gpus.append({
                "id": parts[0], "name": parts[1], "util": parts[2],
                "mem_used": parts[3], "mem_total": parts[4],
                "temp": parts[5], "power": parts[6]
            })
        return gpus
    except Exception:
        return []

def main():
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()
    while True:
        lines = []
        # System Spec
        cpu_count = psutil.cpu_count(logical=True)
        cpu_pct = psutil.cpu_percent(interval=None)
        load1, load5, load15 = psutil.getloadavg()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk_root = psutil.disk_usage('/')
        disk_ssd = psutil.disk_usage('/mnt/ssd1') if os.path.exists('/mnt/ssd1') else None

        lines.append("=" * 80)
        lines.append("                   ★ SERVER SYSTEM SPEC & RESOURCE MONITOR ★")
        lines.append("=" * 80)
        lines.append(f" CPU Usage : [{cpu_pct:5.1f}%] | Cores: {cpu_count} | Load Avg: {load1:.2f}, {load5:.2f}, {load15:.2f}")
        lines.append(f" RAM Memory: {mem.used/1024**3:5.1f} GB / {mem.total/1024**3:5.1f} GB ({mem.percent}%) | Free: {mem.available/1024**3:5.1f} GB")
        lines.append(f" Swap Mem  : {swap.used/1024**3:5.1f} GB / {swap.total/1024**3:5.1f} GB ({swap.percent}%)")
        lines.append(f" Root Disk : {disk_root.used/1024**3:5.1f} GB / {disk_root.total/1024**3:5.1f} GB ({disk_root.percent}%)")
        if disk_ssd:
            lines.append(f" SSD1 Disk : {disk_ssd.used/1024**3:5.1f} GB / {disk_ssd.total/1024**3:5.1f} GB ({disk_ssd.percent}%)")
        lines.append("-" * 80)
        lines.append(" GPU STATUS:")
        gpus = get_gpu_info()
        for g in gpus:
            lines.append(f"  [GPU {g['id']}] {g['name']:<22} | Util: {g['util']:>3}% | VRAM: {float(g['mem_used'])/1024:4.1f}/{float(g['mem_total'])/1024:4.1f} GB | Temp: {g['temp']}C | Pwr: {float(g['power']):3.0f}W")
        lines.append("-" * 80)
        lines.append(" ACTIVE TRAINING TASKS:")
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                cmd = " ".join(p.info['cmdline'] or [])
                if "train_phase1.py" in cmd and "python" in p.info['name']:
                    lines.append(f"  [Phase 1] PID: {p.info['pid']:<7} | CPU: {p.info['cpu_percent']:>5.1f}% | Mem: {p.info['memory_percent']:>4.1f}% | GPU 0 (Loss Branching)")
                elif "train_phase2.py" in cmd and "python" in p.info['name']:
                    lines.append(f"  [Phase 2] PID: {p.info['pid']:<7} | CPU: {p.info['cpu_percent']:>5.1f}% | Mem: {p.info['memory_percent']:>4.1f}% | GPU 1 (Multi-Frame)")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        lines.append("=" * 80)

        # In-place rewrite: Move cursor to home and overwrite lines with \033[K (clear to end of line)
        buf = '\033[H' + '\n'.join([l + '\033[K' for l in lines]) + '\033[J'
        sys.stdout.write(buf)
        sys.stdout.flush()
        time.sleep(1.5)

if __name__ == "__main__":
    main()
