import pandas as pd
import json
import time
import websocket
import threading
from datetime import datetime, timezone
from queue import Queue, Empty
from common.config import load_config
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = load_config()
logger.info("Loaded config: %s", config)

def convert_candle_list_to_df(candle_list):
    """
    Converts a list of candle dictionaries to a pandas DataFrame,
    sets the 'Date' as the index, and ensures correct data types.
    Returns an empty DataFrame if input is empty or invalid.
    """
    if not candle_list:
        logger.warning("Empty candle_list provided to convert_candle_list_to_df.")
        return pd.DataFrame()
    df = pd.DataFrame(candle_list)
    if 'Date' not in df.columns:
        logger.error("No 'Date' column found in candle_list.")
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')
    # Ensure correct data types for plotting and analysis
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' missing in candle_list.")
    return df

class DerivHistoricalDataClient:
    """
    Client for fetching historical and live candlestick data from Deriv WebSocket API.
    Handles batching, deduplication, and DataFrame conversion.
    """
    MAX_CANDLES_PER_BATCH = 1000  # Decreased batch size for reliability

    def __init__(self, symbol, granularity, start_datetime_str, end_datetime_str, timezone_in_minutes=0, api_token=None):
        """
        Initialize the historical data client for Deriv API
        
        Args:
            symbol (str): Symbol code (e.g., 'frxEURUSD')
            granularity (int): Candle timeframe in seconds
            start_datetime_str (str): ISO format datetime string for start date
            end_datetime_str (str): ISO format datetime string for end date
            timezone_in_minutes (int): Timezone offset in minutes
        """
        # Load configuration
        config = load_config()
        
        # Set up class properties
        self.symbol = symbol
        self.granularity = granularity
        self.start_datetime_str = start_datetime_str
        self.end_datetime_str = end_datetime_str
        
        # Make sure these attributes are defined
        self.ws = None
        self.ws_app = None
        self.ws_thread = None
        
        # Parse datetime strings to epoch seconds
        self.start_datetime_s = self.parse_datetime_to_epoch(start_datetime_str)
        self.end_datetime_s = self.parse_datetime_to_epoch(end_datetime_str)
        
        # Additional configuration
        self.timezone_in_minutes = timezone_in_minutes
        self.all_candles = []
        self.api_token = api_token
        self.is_authorized = False
        self._next_on_open = None
        self._post_auth_on_open = None
        self.is_connected = False

        # Set up WebSocket connection properties
        self.ws = None
        self.ws_app = None  # Add this missing attribute
        self.ws_thread = None  # Add this missing attribute
        self.data_ready_event = threading.Event()
        self.tick_queue = Queue()
        
        # Get app_id from config
        app_id = 1089  # Default app_id
        api_url = "wss://ws.binaryws.com/websockets/v3"  # Default URL
        
        if 'mybot' in config:
            app_id = config['mybot'].get('app_id', app_id)
            api_url = config['mybot'].get('deriv_api_url', api_url)
        elif 'deriv_api' in config:
            app_id = config['deriv_api'].get('app_id', app_id)
            # Use default API URL if not specified
            
        self.ws_url = f"{api_url}?app_id={str(app_id)}"
        logger.info(f"WebSocket URL: {self.ws_url}")
        self.all_candle_data = []
        
    def parse_datetime_to_epoch(self, datetime_str):
        """Convert ISO datetime string to epoch seconds"""
        try:
            # Handle ISO format datetime string
            if 'Z' in datetime_str:
                datetime_str = datetime_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except Exception as e:
            print(f"Error parsing datetime {datetime_str}: {e}")
            return int(datetime.now().timestamp())
            

    def _on_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get("msg_type")
        if msg_type == "authorize":
            if data.get("error"):
                logger.error(f"Authorization failed: {data['error']}")
            else:
                self.is_authorized = True
                logger.info("Authorization successful.")
                # After successful auth, proceed with the intended on_open action for this connection
                if self._post_auth_on_open:
                    try:
                        self._post_auth_on_open(ws)
                    except Exception as e:
                        logger.error(f"Error running post-auth on_open handler: {e}")
                    finally:
                        self._post_auth_on_open = None
            return
        if msg_type == "candles":
            is_historical_response = "echo_req" in data and data["echo_req"].get("ticks_history")
            received_candles = data.get("candles", [])
            if received_candles:
                for candle in received_candles:
                    self.candle_data.append({
                        "Date": datetime.fromtimestamp(candle["epoch"], tz=timezone.utc),
                        "Open": float(candle["open"]),
                        "High": float(candle["high"]),
                        "Low": float(candle["low"]),
                        "Close": float(candle["close"]),
                        "Volume": float(candle.get("volume", 0))
                    })
                if is_historical_response:
                    logger.info(f"Received {len(received_candles)} historical candlestick candles (response to request).")
                    self.data_ready_event.set()
                else:
                    logger.info(f"Received {len(received_candles)} live candlestick updates.")
                    for candle in received_candles:
                        self.tick_queue.put({'type': 'candle', 'candle': {
                            "Date": datetime.fromtimestamp(candle["epoch"], tz=timezone.utc),
                            "Open": float(candle["open"]),
                            "High": float(candle["high"]),
                            "Low": float(candle["low"]),
                            "Close": float(candle["close"]),
                            "Volume": float(candle.get("volume", 0))
                        }})
            else:
                if is_historical_response:
                    logger.warning("'candles' message received for history, but candle list is empty. Unblocking.")
                    self.data_ready_event.set()
                else:
                    logger.debug("Received empty 'candles' message (possibly a live update with no data).")
        elif msg_type == "tick":
            tick = data.get("tick")
            if tick:
                epoch_time = tick["epoch"]
                price = float(tick["quote"])
                time_dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
                self.times.append(time_dt)
                self.prices.append(price)
                self.tick_queue.put({'type': 'tick', 'time': time_dt, 'price': price})
        else:
            logger.debug(f"Received unhandled message type: {msg_type}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket Connection closed")
        self.is_connected = False
        self.tick_queue.put(None)

    def _on_open_history_request_batch(self, ws, batch_start_epoch, batch_end_epoch, batch_count):
        logger.info(f"Sending batch request: Symbol={self.symbol}, Granularity={self.granularity}, "
                    f"Start={batch_start_epoch} ({datetime.fromtimestamp(batch_start_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}), "
                    f"End={batch_end_epoch} ({datetime.fromtimestamp(batch_end_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}), Count={batch_count}")
        history_request = {
            "ticks_history": self.symbol,
            "adjust_start_time": 1,
            "granularity": self.granularity,
            "count": batch_count,
            "end": batch_end_epoch,
            "start": batch_start_epoch,
            "style": "candles"
        }
        logger.debug(f"Sending historical batch request: {json.dumps(history_request, indent=2)}")
        ws.send(json.dumps(history_request))
        self.is_connected = True

    def _on_open(self, ws):
        """Internal on_open wrapper to optionally authorize before sending requests."""
        logger.info("WebSocket opened")
        self.is_connected = True
        if self.api_token and not self.is_authorized:
            # Send authorize first, then defer actual on_open work until after auth
            try:
                logger.info("Sending authorization request...")
                ws.send(json.dumps({"authorize": self.api_token}))
                # After authorize response, _on_message will call _post_auth_on_open
                self._post_auth_on_open = self._next_on_open
            except Exception as e:
                logger.error(f"Failed to send authorize: {e}")
        else:
            if self._next_on_open:
                try:
                    self._next_on_open(ws)
                except Exception as e:
                    logger.error(f"Error in on_open handler: {e}")

    def connect(self, on_open_handler=None):
        # Store the desired on_open handler for this connection
        self._next_on_open = on_open_handler
        if self.ws_app is None or not self.ws_thread or not self.ws_thread.is_alive():
            self.ws_app = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            logger.info("WebSocket connection initiated.")
        else:
            logger.info("WebSocket is already connected.")

    def disconnect(self):
        if self.ws_app and self.is_connected:
            self.ws_app.close()
            logger.info("Disconnected from WebSocket.")
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
            logger.info("WebSocket thread joined.")
        self.is_connected = False
        while not self.tick_queue.empty():
            try:
                self.tick_queue.get_nowait()
            except Empty:
                pass

    def get_historical_data(self, timeout=180, max_retries=3, sleep_between_batches=2):
        """
        Retrieve all historical data within the specified time range, handling API chunking and retries.
        Returns: (times, prices, candle_data_list)
        """
        self.all_candle_data = []
        added_epochs = set()
        current_start_epoch = self.start_datetime_s
        original_end_epoch = self.end_datetime_s
        logger.info(f"Requested historical range (Epoch): {current_start_epoch}, {original_end_epoch}")
        loop_iteration = 0

        while current_start_epoch <= original_end_epoch:  # <= to ensure last candle is included
            loop_iteration += 1
            self.data_ready_event.clear()
            self.candle_data = []

            # Calculate how many candles to request in this batch
            remaining_time_delta = original_end_epoch - current_start_epoch
            max_possible_candles_in_range = int(remaining_time_delta / self.granularity)
            batch_count = min(self.MAX_CANDLES_PER_BATCH, max(1, max_possible_candles_in_range) + 1)  # +1 to ensure inclusion
            batch_request_end_epoch = min(original_end_epoch, current_start_epoch + (batch_count - 1) * self.granularity)

            # Defensive: ensure at least one candle is requested
            if remaining_time_delta >= 0 and batch_request_end_epoch < current_start_epoch:
                batch_count = 1
                batch_request_end_epoch = original_end_epoch

            logger.info(f"Requesting batch: Start={datetime.fromtimestamp(current_start_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, "
                        f"End={datetime.fromtimestamp(batch_request_end_epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, Count={batch_count}")

            # --- Retry logic for each batch with forced disconnect/reconnect ---
            for attempt in range(1, max_retries + 1):
                self.disconnect()
                time.sleep(0.5)
                self.connect(on_open_handler=lambda ws: self._on_open_history_request_batch(ws, current_start_epoch, batch_request_end_epoch, batch_count))
                logger.info(f"Waiting for batch data (timeout: {timeout} seconds)... (Attempt {attempt}/{max_retries})")
                if self.data_ready_event.wait(timeout=timeout):
                    break
                else:
                    logger.warning(f"Timeout while waiting for batch historical data (Attempt {attempt}/{max_retries}). Retrying...")
                    self.disconnect()
                    time.sleep(sleep_between_batches)
            else:
                logger.error("Failed to retrieve batch after maximum retries, skipping this batch.")
                # Advance to next batch to avoid infinite loop
                current_start_epoch += (batch_count * self.granularity)
                if current_start_epoch > original_end_epoch:
                    break
                continue

            logger.info(f"Batch data received. Collected {len(self.candle_data)} candles from current batch.")

            if self.candle_data:
                # Deduplicate by epoch
                newly_added_this_batch = []
                for c in self.candle_data:
                    candle_epoch = int(c['Date'].timestamp())
                    if candle_epoch <= original_end_epoch and candle_epoch not in added_epochs:
                        newly_added_this_batch.append(c)
                        added_epochs.add(candle_epoch)
                if not newly_added_this_batch:
                    logger.info("No *new* candles added from this batch after filtering/deduplication. Terminating.")
                    current_start_epoch = int(batch_request_end_epoch + self.granularity)
                    if current_start_epoch > original_end_epoch:
                        break
                    time.sleep(sleep_between_batches)
                    continue
                self.all_candle_data.extend(newly_added_this_batch)
                # Advance to just after the last candle received
                last_dt = newly_added_this_batch[-1]['Date']
                current_start_epoch = int(last_dt.timestamp()) + self.granularity
                if current_start_epoch > original_end_epoch:
                    break
            else:
                logger.warning("Batch returned no data. Advancing current_start_epoch by current batch's expected range to prevent infinite loop.")
                current_start_epoch += (batch_count * self.granularity)
                if current_start_epoch > original_end_epoch:
                    break

            self.disconnect()
            time.sleep(sleep_between_batches)

        # Sort and return
        if len(self.all_candle_data) > 1:
            self.all_candle_data.sort(key=lambda x: x['Date'])
        logger.info(f"Finished collecting all historical data. Total candles: {len(self.all_candle_data)}")
        final_times = [c['Date'] for c in self.all_candle_data]
        final_prices = [c['Close'] for c in self.all_candle_data]
        return final_times, final_prices, self.all_candle_data

    def subscribe_to_live_ticks(self):
        """
        Subscribe to live tick data (requires _on_open_live_subscribe implementation).
        """
        if not hasattr(self, '_on_open_live_subscribe'):
            logger.error("_on_open_live_subscribe handler not implemented.")
            return
        self.connect(on_open_handler=self._on_open_live_subscribe)
        logger.info("Live tick subscription initiated.")

    def get_next_tick(self, timeout=None):
        """
        Get the next tick or candle from the queue, or None on timeout/end.
        """
        try:
            item = self.tick_queue.get(block=True, timeout=timeout)
            return item
        except Exception:
            return None

    def get_all_candle_data_df(self):
        """
        Returns the accumulated candle data as a pandas DataFrame.
        """
        logger.info(f"Inside get_all_candle_data_df: Length of self.all_candle_data = {len(self.all_candle_data)}")
        if not self.all_candle_data:
            logger.warning("self.all_candle_data is empty, returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.DataFrame(self.all_candle_data)
        logger.debug(f"DataFrame created from self.all_candle_data. Head:\n{df.head()}")
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            logger.debug(f"After to_datetime. Date column dtypes: {df['Date'].dtype}")
            if df['Date'].isnull().all():
                logger.warning("All 'Date' values coerced to NaT. Check date format in self.all_candle_data.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error converting 'Date' column: {e}")
            return pd.DataFrame()
        df = df.set_index('Date')
        logger.debug(f"After set_index('Date'). Index type: {df.index.dtype}")
        try:
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float})
            logger.debug(f"After astype. DataFrame dtypes:\n{df.dtypes}")
        except Exception as e:
            logger.error(f"Error converting price columns to float: {e}")
            return pd.DataFrame()
        logger.info(f"Final DataFrame shape: {df.shape}")
        return df

"""
### Data Retrieval Timeout Troubleshooting

If you see a timeout like:

    WARNING:common.data_handler:Timeout while waiting for batch historical data. Disconnecting.

**Possible causes and solutions:**

1. **Batch Size Too Large**  
   - The API or network may struggle to deliver 5000 candles in one go.
   - Try reducing `MAX_CANDLES_PER_BATCH` (e.g., to 1000 or 500).

2. **Timeout Too Short**  
   - If your network is slow or the API is under load, increase the `timeout` parameter (e.g., to 180 or 300 seconds).

3. **Network Instability**  
   - Ensure your internet connection is stable.
   - If using a VPN or proxy, try disabling it.

4. **API Rate Limits or Server Issues**  
   - If you hit rate limits, add a longer `time.sleep()` between batches (e.g., 1-2 seconds).
   - If the API is unreliable, consider retrying failed batches a few times before giving up.

5. **WebSocket Connection Handling**  
   - If you see "WebSocket is already connected", ensure you are not reusing a stale connection.  
   - You may want to force a disconnect and reconnect for each batch.

**Recommended tweaks:**

- Lower `MAX_CANDLES_PER_BATCH` to 1000 or 500.
- Increase `timeout` to 180 or 300.
- Add a retry loop for failed batches (try up to 3 times before skipping).
- Add a longer sleep (e.g., `time.sleep(1)`) between batches.

**Example code changes:**

```python
# In DerivHistoricalDataClient:
MAX_CANDLES_PER_BATCH = 1000  # or 500

# In get_historical_data, after self.disconnect():
time.sleep(1)  # instead of 0.5

# Add retry logic for each batch:
for attempt in range(3):
    self.connect(...)
    if self.data_ready_event.wait(timeout=timeout):
        break
    else:
        logger.warning(f"Timeout on attempt {attempt+1}, retrying...")
        self.disconnect()
        time.sleep(2)
else:
    logger.error("Failed to retrieve batch after 3 attempts, skipping this batch.")
    break
```

**Summary:**  
- Lower batch size, increase timeout, add retries, and ensure clean WebSocket reconnects for each batch.
"""