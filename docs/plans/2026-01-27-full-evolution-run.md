# Full Evolution Run Plan - STaRK Prime

## Goal

Achieve the following metrics on STaRK Prime dataset:
- Hit@1 > 50%
- Hit@5 > 75%
- Recall@20 > 80%
- MRR > 75%

## Run Commands

### Phase 1: Weighted Sum (Start Here)

```bash
cd stark
python evolve_stark.py --dataset prime --mode weighted_sum --gens 500 --train_ss 100 --val_ss 50
```

### Phase 2: Expression Tree (If Phase 1 Plateaus)

```bash
python evolve_stark.py --dataset prime --mode expression_tree --gens 300 --train_ss 100 --val_ss 50
```

### Resume from Checkpoint

```bash
python evolve_stark.py --dataset prime --resume runs/prime/YYYYMMDD_HHMMSS
```

## Monitoring

Watch for:
- Training fitness trajectory (steady improvement)
- Validation fitness (should track training)
- Archive coverage (target >40%)
- Tier exit distribution (healthy: 60/20/15/5)

## Early Stop Conditions

1. **Goal reached** - All 4 metrics hit targets on validation
2. **Plateau** - Best fitness unchanged for 50+ generations
3. **Overfitting** - Training improving but validation declining

## Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Hit@1 | 40% | 50% | 55% |
| Hit@5 | 65% | 75% | 80% |
| Recall@20 | 70% | 80% | 85% |
| MRR | 60% | 75% | 80% |
