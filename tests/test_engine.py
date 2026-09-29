import unittest
from src.engine.order_book import OrderBook, Order, Side, OrderType, Status
from src.engine.matching_engine import MatchingEngine

class TestMatchingEngine(unittest.TestCase):
    def setUp(self):
        self.order_book = OrderBook()
        self.engine = MatchingEngine(self.order_book)
        
    def test_submit_valid_limit_order(self):
        order = Order(
            order_id="o1",
            agent_id="a1",
            side=Side.BID,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10,
            timestamp_step=1
        )
        trades = self.engine.process(order)
        self.assertEqual(len(trades), 0)
        
        best_bid = self.order_book.best_bid()
        self.assertIsNotNone(best_bid)
        self.assertEqual(best_bid.price, 100.0)
        self.assertEqual(best_bid.quantity, 10)
        
    def test_crossing_orders_match(self):
        bid_order = Order(
            order_id="o1", agent_id="a1", side=Side.BID,
            order_type=OrderType.LIMIT, price=100.0, quantity=10, timestamp_step=1
        )
        self.engine.process(bid_order)
        
        ask_order = Order(
            order_id="o2", agent_id="a2", side=Side.ASK,
            order_type=OrderType.LIMIT, price=99.0, quantity=5, timestamp_step=2
        )
        trades = self.engine.process(ask_order)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 5)
        self.assertEqual(trades[0].price, 100.0) # Matches at resting order's price
        
        self.assertEqual(bid_order.quantity, 5)
        self.assertEqual(bid_order.status, Status.PARTIALLY_FILLED)
        
        self.assertEqual(ask_order.quantity, 0)
        self.assertEqual(ask_order.status, Status.FILLED)

if __name__ == "__main__":
    unittest.main()
