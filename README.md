# orderbook-amm-hybrid-sim
Empirical comparison of price discovery, capital efficiency, and rent extraction across AMM, CLOB, &amp; hybrid market-making regimes under varying agent populations.

# Mechanism Comparison Sandbox

An empirical comparison of price discovery, capital efficiency, and rent extraction
across AMM, CLOB, and hybrid market-making regimes.

## Context

This work builds on prior simulation infrastructure I developed studying LS-LMSR-
based prediction markets for event-triggered RWAs ([prior repo](link to lmsr-
preclinical-markets)). After reading Phoenix-v1 and Plasma, I wanted to understand
which of those findings translate to CLOB venues and where the hybrid orderbook/AMM
pattern materially changes mechanism behavior.

This is exploration of the design space, not a model of any specific deployed
venue.

## What's here

- Three minimal venue implementations: a constant-product AMM, a stylized CLOB,
  and a hybrid CLOB seeded with passive programmatic quotes
- Four agent classes ported from prior work: informed, noise, momentum, adversarial
- A sweep harness that runs the same agent populations against each venue
- Metrics: price convergence, rent efficiency, capital saturation, slippage at size

## Findings

[Three to five paragraphs of headline observations, with embedded chart references]

## Limitations

[Explicit hedges]

## What's next

[The Option C natural extension + other directions]
