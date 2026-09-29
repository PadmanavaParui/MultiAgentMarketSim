# MultiAgentMarketSim
Multi Agent Market Simulation
## how to run it
Open terminal and run the following commands:
```bash
cd /home/tw1light/Desktop/mams
python3 scripts/demo_matching_engine.py
```

expected output:
```bash
Initializing Multi-Agent Market Simulation - Matching Engine Demo...

[1] Market Makers are placing initial Limit Orders...

=== Current Order Book Depth ===
BID (Buy)            | ASK (Sell)          
---------------------------------------------
100 shares @ $99.00  | 100 shares @ $101.00
200 shares @ $98.50  | 150 shares @ $101.50
================================

[2] Retail Agent submits an aggressive BUY Limit Order (Qty: 120, Price: $101.50)...

[3] Trades Executed:
 -> Trade ID: trd_1 | Matched 100 shares at $101.00
 -> Trade ID: trd_2 | Matched 20 shares at $101.50

[4] Order Book state after trades:

=== Current Order Book Depth ===
BID (Buy)            | ASK (Sell)          
---------------------------------------------
100 shares @ $99.00  | 130 shares @ $101.50
200 shares @ $98.50  |                     
================================

Notice how the $101.00 Ask level was completely consumed, and the $101.50 Ask level was partially filled!
```

