# From Theory to Pitch Chapter 2: Space and dominance in the 2022 World Cup (Repository)

This repository contains the implementation of the analysis presented in the article:

**[“From Theory to Pitch Chapter 2: Space and dominance in the 2022 World Cup”](https://open.substack.com/pub/gabrielgausachs/p/from-theory-to-pitch-chapter-2-space?r=884tdi&utm_campaign=post-expanded-share&utm_medium=web)**

---

## Overview

This project explores how space can be quantified in football using Pitch Control, Pitch Value, and Space Quality, inspired by the paper **“Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer”** by Fernández and Bornn et al. (2018).

The goal is to bring these concepts to life using data from the 2022 FIFA World Cup and show how they can be used to analyse whether teams move the ball into more valuable areas of the pitch after recovering possession.

---

## Data

The analysis is based on tracking and event data from the 2022 FIFA World Cup, processed and cleaned for consistency and analytical use.

The project uses:

- **Pitch control** computed with `DataballPy`
- A custom **pitch value model** trained following the idea proposed in *Wide Open Spaces*
- **Space quality**, computed by combining pitch control and pitch value

The main analysis focuses on pass-only recoveries from the final knockout rounds of the tournament, using this smaller sample to clearly demonstrate what can be done with pitch control, pitch value, and space quality without making the computation unnecessarily expensive.

---

## Acknowledgements

Special thanks to the authors of the original space creation framework:

- Fernández and Bornn et al. (2018), *Wide Open Spaces: A Statistical Technique for Measuring Space Creation in Professional Soccer*

And to:

- Martens et al. (2021), *Space and Control in Soccer*
- Wu and Swartz et al. (2024), *A new metric for pitch control based on an intuitive motion model*
- Karun Singh, *Introducing Expected Threat (xT)*

- PFF FC for making the 2022 World Cup tracking and event data publicly available
- DataballPy for providing the pitch control implementation used in this work


---

## Note

This repository serves as the implementation accompanying the article and is part of a personal exploration into football analytics.

Hope you enjoy it — any feedback or corrections are more than welcome.

