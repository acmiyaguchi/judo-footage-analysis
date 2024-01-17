# acmiyaguchi

This is a directory for me to throw in random scripts or docs that may not belong elsewhere related to analysis.

```bash
./user/acmiyaguchi/run_detectron.py \
    --input data/interim/mat-2-trunc.mp4 \
    --output data/processed/mat-2-viz.mp4 \
    --output-data-root data/processed/mat-2-data \
    --duration 5
```

```bash
python3 -m b2 sync data/processed b2://acm-judo/data/analysis/processed
```
