#!/bin/bash

while true
do
    echo "=== 啟動 RL 訓練 ==="
    python train_high_level.py

    echo "=== RL 訓練已結束，10秒後重啟 ==="
    sleep 10
done
