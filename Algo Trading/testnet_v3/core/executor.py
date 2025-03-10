"""
Order executor to place and manage trades on Binance Futures.
"""
import logging
import time
from typing import Dict, List, Any, Optional

class OrderExecutor:
    """
    Handles placing and managing orders on Binance Futures.
    """
    
    def __init__(self, client, symbols: List[str], test_mode: bool = True, logger=None):
        """
        Initialize the order executor.
        
        Args:
            client: Binance Futures client
            symbols: List of trading symbols
            test_mode: Whether to use test mode (no real orders placed)
            logger: Optional logger instance
        """
        self.client = client
        self.symbols = symbols
        self.test_mode = test_mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Track open orders
        self.open_orders = {}
        
    def execute_trades(self, trade_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute trade decisions by placing orders on Binance.
        
        Args:
            trade_decisions: Dictionary of trade decisions from portfolio manager
                Format: {
                    symbol: {
                        "side": "BUY"/"SELL"/"CLOSE",
                        "quantity": float,
                        "price": float,
                        "type": "MARKET"/"LIMIT"
                    }
                }
                
        Returns:
            Dictionary of order results
        """
        results = {}
        
        if not trade_decisions:
            self.logger.info("No trades to execute")
            return results
            
        self.logger.info(f"Executing {len(trade_decisions)} trade decisions")
        
        # First handle any position closures
        for symbol, decision in trade_decisions.items():
            if decision['side'] == 'CLOSE':
                result = self._close_position(symbol, decision)
                results[f"{symbol}_CLOSE"] = result
        
        # Then handle new positions
        for symbol, decision in trade_decisions.items():
            if decision['side'] in ['BUY', 'SELL']:
                result = self._place_order(symbol, decision)
                results[symbol] = result
                
        return results
        
    def _place_order(self, symbol: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a single order.
        
        Args:
            symbol: Trading symbol
            decision: Trade decision dictionary
            
        Returns:
            Order result dictionary
        """
        try:
            side = decision['side']
            quantity = decision['quantity']
            order_type = decision['type']
            
            self.logger.info(f"Placing {side} {order_type} order for {quantity} {symbol}")
            
            if self.test_mode:
                # Simulate order in test mode
                order_id = f"test_{int(time.time() * 1000)}"
                order_result = {
                    'orderId': order_id,
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity,
                    'status': 'FILLED',  # Simulate immediate fill in test mode
                    'test': True
                }
                self.logger.info(f"TEST MODE: Simulated order {order_id} for {symbol}")
            else:
                # Place real order
                if order_type == 'MARKET':
                    order_result = self.client.new_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                        reduceOnly=False,
                        newOrderRespType="RESULT"  # To get fill information
                    )
                elif order_type == 'LIMIT':
                    price = decision['price']
                    order_result = self.client.new_order(
                        symbol=symbol,
                        side=side,
                        type="LIMIT",
                        quantity=quantity,
                        price=price,
                        timeInForce="GTC",
                        reduceOnly=False,
                        newOrderRespType="RESULT"
                    )
                
                self.logger.info(f"Order placed for {symbol}: ID {order_result.get('orderId')}, Status: {order_result.get('status')}")
                
                # Track open order if not immediately filled
                if order_result.get('status') != 'FILLED':
                    self.open_orders[order_result['orderId']] = order_result
            
            return order_result
            
        except Exception as e:
            error_msg = f"Error placing order for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'symbol': symbol, 'status': 'FAILED'}
    
    def _close_position(self, symbol: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            decision: Trade decision dictionary
            
        Returns:
            Order result dictionary
        """
        try:
            quantity = decision['quantity']
            # Determine the side needed to close the position
            # If we're closing a long position, we need to sell
            # If we're closing a short position, we need to buy
            close_side = 'SELL'  # Default assumption for closing long position
            
            # In a real implementation, you'd check the actual position side
            # from self.client.get_position_risk(symbol=symbol)
            
            self.logger.info(f"Closing position for {symbol}, quantity: {quantity}")
            
            if self.test_mode:
                # Simulate order in test mode
                order_id = f"test_close_{int(time.time() * 1000)}"
                order_result = {
                    'orderId': order_id,
                    'symbol': symbol,
                    'side': close_side,
                    'type': 'MARKET',
                    'quantity': quantity,
                    'status': 'FILLED',
                    'reduceOnly': True,
                    'test': True
                }
                self.logger.info(f"TEST MODE: Simulated closing order {order_id} for {symbol}")
            else:
                # Place real order with reduceOnly flag
                order_result = self.client.new_order(
                    symbol=symbol,
                    side=close_side,
                    type="MARKET",
                    quantity=quantity,
                    reduceOnly=True,
                    newOrderRespType="RESULT"
                )
                
                self.logger.info(f"Position close order for {symbol}: ID {order_result.get('orderId')}, Status: {order_result.get('status')}")
            
            return order_result
            
        except Exception as e:
            error_msg = f"Error closing position for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'symbol': symbol, 'status': 'FAILED'}
    
    def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Check the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Order status dictionary
        """
        try:
            if self.test_mode and order_id.startswith('test_'):
                # Simulate in test mode
                return {'orderId': order_id, 'status': 'FILLED', 'test': True}
            
            order_status = self.client.query_order(orderId=order_id)
            
            # If order is no longer open, remove from tracking
            if order_status.get('status') not in ['NEW', 'PARTIALLY_FILLED']:
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
            
            return order_status
            
        except Exception as e:
            self.logger.error(f"Error checking order status for {order_id}: {str(e)}")
            return {'orderId': order_id, 'error': str(e), 'status': 'UNKNOWN'}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Cancellation result dictionary
        """
        try:
            if self.test_mode and order_id.startswith('test_'):
                # Simulate in test mode
                return {'orderId': order_id, 'status': 'CANCELED', 'test': True}
            
            cancel_result = self.client.cancel_order(orderId=order_id)
            
            # Remove from tracking
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            
            return cancel_result
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return {'orderId': order_id, 'error': str(e), 'status': 'FAILED_TO_CANCEL'}
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all open orders, optionally for a specific symbol.
        
        Args:
            symbol: Optional symbol to cancel orders for
            
        Returns:
            List of cancellation results
        """
        results = []
        
        try:
            if symbol:
                if self.test_mode:
                    # Simulate in test mode
                    return [{'symbol': symbol, 'status': 'CANCELED', 'test': True}]
                
                cancel_result = self.client.cancel_all_orders(symbol=symbol)
                results.append(cancel_result)
            else:
                if self.test_mode:
                    # Simulate in test mode
                    return [{'allOrdersCanceled': True, 'test': True}]
                
                # Cancel orders for each symbol
                for sym in self.symbols:
                    try:
                        cancel_result = self.client.cancel_all_orders(symbol=sym)
                        results.append(cancel_result)
                    except Exception as e:
                        self.logger.error(f"Error canceling orders for {sym}: {str(e)}")
                        results.append({'symbol': sym, 'error': str(e)})
            
            # Clear tracking
            self.open_orders = {}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {str(e)}")
            return [{'error': str(e), 'status': 'FAILED_TO_CANCEL'}] 