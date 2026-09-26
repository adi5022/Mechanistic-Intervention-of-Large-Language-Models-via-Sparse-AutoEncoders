**Per-arm results, all mode**

| Arm | Prompts | Reached rank #1 | Success rate % | Mean rank gain | Median final rank | Mean final prob % | Features at best | Rescued in best | Steps | Time s | Collateral KL (nats) | Top-1 flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | 6 | 5 | 83.3 | 3.1667 | 1 | 10.565 | 39.0 | 0.0 | 20.5 | 2.416 | 0.0017 | 0.0167 |
| off | 6 | 4 | 66.7 | 2.6667 | 1 | 8.4439 | 51.6667 | 0.0 | 45.3333 | 3.1642 | 0.0029 | 0.0333 |
| tol5_after | 6 | 5 | 83.3 | 3.1667 | 1 | 10.5247 | 36.6667 | 0.0 | 20.6667 | 2.2105 | 0.0015 | 0.0167 |
| graded5_after | 6 | 5 | 83.3 | 3.1667 | 1 | 10.5247 | 36.6667 | 0.0 | 20.6667 | 2.2532 | 0.0015 | 0.0167 |
| graded5_inter | 6 | 5 | 83.3 | 3.1667 | 1 | 10.4855 | 38.3333 | 0.5 | 20.6667 | 2.3933 | 0.0016 | 0.0167 |
