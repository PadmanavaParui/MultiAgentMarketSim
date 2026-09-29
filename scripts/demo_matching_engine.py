import sys
import os

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.order_book import OrderBook, Order, Side, OrderType
from src.engine.matching_engine import MatchingEngine

def print_depth(order_book: OrderBook):
    depth = order_book.depth(levels=3)
    print("\n=== Current Order Book Depth ===")
    print(f"{'BID (Buy)':<20} | {'ASK (Sell)':<20}")
    print("-" * 45)
    
    bids = depth["bids"]
    asks = depth["asks"]
    
    max_len = max(len(bids), len(asks))
    for i in range(max_len):
        bid_str = f"{bids[i][1]} shares @ ${bids[i][0]:.2f}" if i < len(bids) else ""
        ask_str = f"{asks[i][1]} shares @ ${asks[i][0]:.2f}" if i < len(asks) else ""
        print(f"{bid_str:<20} | {ask_str:<20}")
    print("================================\n")

def run_demo():
    print("Initializing Multi-Agent Market Simulation - Matching Engine Demo...\n")
    
    book = OrderBook()
    engine = MatchingEngine(book)
    
    # 1. Setup an initial resting order book (Liquidity provision)
    print("[1] Market Makers are placing initial Limit Orders...")
    initial_orders = [
        Order("order_1", "market_maker_1", Side.BID, OrderType.LIMIT, 99.00, 100, 1),
        Order("order_2", "market_maker_1", Side.BID, OrderType.LIMIT, 98.50, 200, 2),
        Order("order_3", "market_maker_2", Side.ASK, OrderType.LIMIT, 101.00, 100, 3),
        Order("order_4", "market_maker_2", Side.ASK, OrderType.LIMIT, 101.50, 150, 4)
    ]
    
    for order in initial_orders:
        engine.process(order)
        
    print_depth(book)
    
    # 2. A retail trader submits a large buy order that crosses the spread
    print("[2] Retail Agent submits an aggressive BUY Limit Order (Qty: 120, Price: $101.50)...")
    aggressive_buy = Order("order_5", "retail_agent_1", Side.BID, OrderType.LIMIT, 101.50, 120, 5)
    
    trades = engine.process(aggressive_buy)
    
    # 3. Show the resulting trades
    print("\n[3] Trades Executed:")
    if not trades:
        print("No trades executed.")
    for trade in trades:
        print(f" -> Trade ID: {trade.trade_id} | Matched {trade.quantity} shares at ${trade.price:.2f}")
        
    # 4. Show the final order book state
    print("\n[4] Order Book state after trades:")
    print_depth(book)
    print("Notice how the $101.00 Ask level was completely consumed, and the $101.50 Ask level was partially filled!\n")

if __name__ == "__main__":
    run_demo()
