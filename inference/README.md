# PaperTrading

## Running bot
```
python inference/main.py \
    --model-file "models/model-LogisticRegression-p1-tr8-sw2-lw42-fa0.5-max100000-fee_flat0-fee_pct0.001-#0.sav" \
    --scaler-file "models/x-scaler-LogisticRegression-p1-tr8-sw2-lw42-fa0.5-max100000-fee_flat0-fee_pct0.001-#0.sav" \
    --wallet-id 12345 \
    --txn-max 100000 \
    --slow_sma_window 42
```