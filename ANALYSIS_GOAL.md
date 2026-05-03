
# Main Analysis: Space Exploitation After Recovery

## Main Idea

The goal is to measure:

> After a team recovers the ball, how well do they transform that recovery into higher-quality space?

Instead of only counting recoveries, this analysis evaluates the **spatial value created after the recovery**.

For every recovery:

1. Compute the space quality around the ball at the recovery time.
2. Track the space quality around the ball for the next 5 seconds.
3. Measure how much the team improves its spatial situation.

The main metric is:

$$
RecoverySpaceGain = \max(SQ_{t_0:t_0+5s}) - SQ_{t_0}
$$

Where:

- $SQ_{t_0}$ is the space quality around the ball at the recovery moment.
- $\max(SQ_{t_0:t_0+5s})$ is the maximum space quality reached in the 5 seconds after recovery.

---

## Plot 1: Team Ranking — Average Recovery Space Gain

### Question

> Which teams improve their space quality most after winning the ball?

For each team, calculate the average:

$$
\max(SQ_{next\ 5s}) - SQ_{recovery}
$$

### What It Shows

This gives a clean team ranking of **transition space exploitation**.

A high value means:

> This team often turns recoveries into better spatial situations.

A low value means:

> This team recovers the ball but does not usually improve the spatial situation quickly.

---

## Plot 2: Space Quality Curve After Recovery

### Question

> How quickly do teams reach better space after recovering the ball?

For each recovery, track the average space quality at:

- 0s
- 1s
- 2s
- 3s
- 4s
- 5s

Then average the curves by team.

### What It Shows

This shows the **tempo of space exploitation** after recovery.

Possible interpretations:

- **Sharp early increase:** fast transition team
- **Slow increase:** patient progression
- **Flat curve:** conservative or ineffective transition
- **Increase then drop:** team finds space but cannot maintain it

This is probably the strongest visual plot in the analysis.

---

## Plot 3: Positive Exploitation Rate

### Question

> How often does each team actually improve space after recovery?

For each recovery, check whether:

$$
RecoverySpaceGain > 0
$$

or, using a stronger condition:

$$
RecoverySpaceGain > threshold
$$

Then compute the percentage of recoveries with positive space gain.

### What It Shows

Plot 1 tells you the **average size** of the gain.

This plot tells you the **consistency** of the team.

A team could have:

- Few recoveries with huge gains
- Many recoveries with small positive gains
- Many recoveries but mostly no improvement

So this plot complements the average recovery space gain ranking.
