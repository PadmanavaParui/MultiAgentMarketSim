from src.engine.order_book import OrderBook

class SimulationInvalidError(Exception):
    def __init__(self, message: str, run_id: str, code: str = "SIMULATION_INVALID"):
        super().__init__(message)
        self.run_id = run_id
        self.code = code

def validate_no_cross(order_book: OrderBook, run_id: str):
    """
    Validates that the best bid is strictly lower than the best ask.
    Raises SimulationInvalidError if the spread is crossed.
    """
    best_bid = order_book.best_bid()
    best_ask = order_book.best_ask()
    
    if best_bid and best_ask:
        if best_bid.price >= best_ask.price:
            raise SimulationInvalidError(
                message=f"Spread-crossing detected! Best bid: {best_bid.price}, Best ask: {best_ask.price}",
                run_id=run_id
            )

def validate_zero_sum(agents: dict, initial_total_value: float, current_total_value: float, fees_collected: float, run_id: str):
    """
    Validates that the total value (fiat + asset) across all agents remains constant, 
    accounting for any explicitly modeled fees.
    """
    expected_value = initial_total_value - fees_collected
    
    # We allow a small float tolerance for zero-sum check
    if abs(expected_value - current_total_value) > 1e-6:
        raise SimulationInvalidError(
            message=f"Zero-sum inventory violation! Expected: {expected_value}, Actual: {current_total_value}",
            run_id=run_id
        )
