Found a genome that has:

Hit@1 > 60%
Hit@5 > 80%
Recall@20 > 85%
MRR > 80%

across the test set and / or the validation set and / or the human eval set.

for all three graphs, stark prime, amazon, and mag.

Main focus is stark prime

The evolution loop should be able to run 500 gens in 3 hours or less with a population of 50-100. 

The current query speed is 50ms per query for stark prime. 

Hopeful goal: have these latency goals still met for not only the weighted sum based evolution, but also the expression trees.

Memory should no exceed a few gbs of VRAM for gpu evolution runs (even for amazon and mag, which are ~10x bigger than prime)