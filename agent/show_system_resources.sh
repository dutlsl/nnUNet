#!/bin/bash
echo "==================== [GPU RESOURCE STATUS] ===================="
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv | column -t -s ","

echo ""
echo "==================== [CPU OVERALL USAGE & LOAD] ===================="
/home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/python -c "
import psutil
cpu_pct = psutil.cpu_percent(interval=0.1)
cores = psutil.cpu_count()
l1, l5, l15 = psutil.getloadavg()
print(f'Total CPU Usage : {cpu_pct:>5.1f}% (Total {cores} Cores)')
print(f'System Load Avg : {l1:.2f} (1m), {l5:.2f} (5m), {l15:.2f} (15m)')
"

echo ""
echo "==================== [TOP CPU PROCESSES] ===================="
ps -eo pid,%cpu,%mem,cmd --sort=-%cpu | head -n 6

echo ""
echo "==================== [SYSTEM MEMORY (RAM)] ===================="
free -h

echo ""
echo "==================== [STORAGE DISK USAGE] ===================="
df -h / /mnt/ssd1
