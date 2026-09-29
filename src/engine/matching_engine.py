from typing import List, Optional
from dataclasses import dataclass
from src.engine.order_book import OrderBook, Order, Side, OrderType, Status

@dataclass
class Trade:
    trade_id: str
    run_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    timestamp_step: int

class MatchingEngine:
    """
    Processes incoming orders against the OrderBook, matching them based on
    price-time priority.
    """
    def __init__(self, order_book: OrderBook):
        self.order_book = order_book
        self.trades_history: List[Trade] = []
        self._trade_counter = 0

    def process(self, order: Order) -> List[Trade]:
        """
        Main entry point for processing a single order.
        Returns a list of Trade objects resulting from the order.
        """
        trades = []
        
        # Immediate match logic for Market Orders or Limit Orders crossing the spread
        if order.side == Side.BID:
            trades.extend(self._match_bid(order))
        else:
            trades.extend(self._match_ask(order))
            
        # If the order is still open and it's a Limit order, add it to the book
        if order.order_type == OrderType.LIMIT and order.quantity > 0:
            self.order_book.add_order(order)
        elif order.order_type == OrderType.MARKET and order.quantity > 0:
            # Unfilled market order is cancelled
            order.status = Status.CANCELLED
            
        if order.quantity == 0:
            order.status = Status.FILLED
            
        return trades

    def _match_bid(self, bid_order: Order) -> List[Trade]:
        trades = []
        while bid_order.quantity > 0:
            best_ask = self.order_book.best_ask()
            
            if not best_ask:
                break
                
            # If it's a limit order, check if prices match
            if bid_order.order_type == OrderType.LIMIT and bid_order.price < best_ask.price:
                break
                
            trade_qty = min(bid_order.quantity, best_ask.quantity)
            trade_price = best_ask.price # Executed at the resting order's price
            
            # Execute Trade
            trade = self._execute_trade(
                buy_order=bid_order, 
                sell_order=best_ask, 
                price=trade_price, 
                quantity=trade_qty
            )
            trades.append(trade)
            
        return trades
        
    def _match_ask(self, ask_order: Order) -> List[Trade]:
        trades = []
        while ask_order.quantity > 0:
            best_bid = self.order_book.best_bid()
            
            if not best_bid:
                break
                
            # If it's a limit order, check if prices match
            if ask_order.order_type == OrderType.LIMIT and ask_order.price > best_bid.price:
                break
                
            trade_qty = min(ask_order.quantity, best_bid.quantity)
            trade_price = best_bid.price # Executed at the resting order's price
            
            # Execute Trade
            trade = self._execute_trade(
                buy_order=best_bid, 
                sell_order=ask_order, 
                price=trade_price, 
                quantity=trade_qty
            )
            trades.append(trade)
            
        return trades
        
    def _execute_trade(self, buy_order: Order, sell_order: Order, price: float, quantity: int) -> Trade:
        self._trade_counter += 1
        trade_id = f"trd_{self._trade_counter}"
        
        buy_order.quantity -= quantity
        sell_order.quantity -= quantity
        
        if buy_order.quantity == 0:
            buy_order.status = Status.FILLED
        elif buy_order.status == Status.OPEN:
            buy_order.status = Status.PARTIALLY_FILLED
            
        if sell_order.quantity == 0:
            sell_order.status = Status.FILLED
        elif sell_order.status == Status.OPEN:
            sell_order.status = Status.PARTIALLY_FILLED
            
        trade = Trade(
            trade_id=trade_id,
            run_id=buy_order.run_id,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            price=price,
            quantity=quantity,
            timestamp_step=max(buy_order.timestamp_step, sell_order.timestamp_step)
        )
        self.trades_history.append(trade)
        return trade
