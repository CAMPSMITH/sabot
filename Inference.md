# PaperTrading

## Running bot
```
python main.py \
    --model-file "models/model-LogisticRegression-p1-tr6-sw4-lw24-fa0.25-max100000-fee_flat14-fee_pct0.0004-#0.sav" \
    --scaler-file "models/x-scaler-LogisticRegression-p1-tr6-sw4-lw24-fa0.25-max100000-fee_flat14-fee_pct0.0004-#0.sav" \
    --wallet-id 12345 \
    --txn-max 100000 \
    --fast_sma_window 4 \
    --slow_sma_window 42
```