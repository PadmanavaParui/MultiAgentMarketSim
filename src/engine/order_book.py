import heapq
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

class Side(Enum):
    BID = 1
    ASK = 2

class OrderType(Enum):
    LIMIT = 1
    MARKET = 2

class Status(Enum):
    OPEN = 1
    PARTIALLY_FILLED = 2
    FILLED = 3
    CANCELLED = 4

@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    timestamp_step: int
    status: Status = Status.OPEN
    run_id: str = "default_run"
    
    def __lt__(self, other):
        # Used as tie-breaker if price and timestamp are the same.
        # Fallback to order_id comparison.
        return self.order_id < other.order_id

class OrderBook:
    """
    Central Limit Order Book (CLOB) using dual priority queues for O(log n) insertion 
    and O(1) best-price retrieval.
    """
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        # Bids: max heap (we negate prices to use Python's min-heap as a max-heap)
        self.bids: List[Tuple[float, int, Order]] = [] 
        # Asks: min heap
        self.asks: List[Tuple[float, int, Order]] = [] 
        self.orders: Dict[str, Order] = {}
        
    def add_order(self, order: Order):
        self.orders[order.order_id] = order
        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BID:
                # Store -price so highest price is at top of min-heap
                heapq.heappush(self.bids, (-order.price, order.timestamp_step, order))
            else:
                heapq.heappush(self.asks, (order.price, order.timestamp_step, order))
                
    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = Status.CANCELLED
            # Note: We rely on lazy deletion in best_bid/best_ask to remove cancelled orders from heaps.
            
    def best_bid(self) -> Optional[Order]:
        """Returns the valid bid with the highest price."""
        while self.bids:
            neg_price, ts, order = self.bids[0]
            if order.status in [Status.CANCELLED, Status.FILLED]:
                heapq.heappop(self.bids)
            else:
                return order
        return None
        
    def best_ask(self) -> Optional[Order]:
        """Returns the valid ask with the lowest price."""
        while self.asks:
            price, ts, order = self.asks[0]
            if order.status in [Status.CANCELLED, Status.FILLED]:
                heapq.heappop(self.asks)
            else:
                return order
        return None
        
    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """Returns a snapshot of the current order book depth."""
        # This is a simplified O(N log N) depth method. Can be optimized if needed.
        # It aggregates volume by price level.
        bids_depth = {}
        for _, _, order in self.bids:
            if order.status not in [Status.CANCELLED, Status.FILLED]:
                bids_depth[order.price] = bids_depth.get(order.price, 0) + order.quantity
                
        asks_depth = {}
        for _, _, order in self.asks:
            if order.status not in [Status.CANCELLED, Status.FILLED]:
                asks_depth[order.price] = asks_depth.get(order.price, 0) + order.quantity
                
        sorted_bids = sorted(bids_depth.items(), key=lambda x: -x[0])[:levels]
        sorted_asks = sorted(asks_depth.items(), key=lambda x: x[0])[:levels]
        
        return {"bids": sorted_bids, "asks": sorted_asks}
