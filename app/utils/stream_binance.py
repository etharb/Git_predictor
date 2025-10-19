import asyncio
import json
import time
import math
from collections import defaultdict, deque
from typing import AsyncIterator, Dict, Any

import websockets

BINANCE_WS = "wss://stream.binance.com:9443/ws"

def sec(ts_ms: int) -> int:
    return ts_ms // 1000

async def trades_stream(symbol: str) -> AsyncIterator[Dict[str, Any]]:
    """Yield trade messages for SYMBOL from Binance spot websocket.
    Message fields used: price, qty, timestamp(ms), is_buyer_maker.
    """
    stream = f"{symbol.lower()}@trade"
    url = f"{BINANCE_WS}/{stream}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=10, ping_timeout=10) as ws:
                async for msg in ws:
                    j = json.loads(msg)
                    # Binance trade payload
                    # { 'p': price str, 'q': qty str, 'T': trade time ms, 'm': is buyer the market maker (bool) }
                    if 'p' in j and 'q' in j and 'T' in j:
                        yield {
                            'price': float(j['p']),
                            'qty': float(j['q']),
                            'ts_ms': int(j['T']),
                            'is_bm': bool(j.get('m', False)),
                        }
        except Exception as e:
            # reconnect after short delay
            await asyncio.sleep(1.0)

async def ohlcv_1s_aggregator(symbol: str, lookback_seconds: int = 1200) -> AsyncIterator[Dict[str, Any]]:
    """Aggregate trades into 1-second OHLCV and buy/sell volume breakdown.
    Yields finalized bars as dicts with keys: ts, open, high, low, close, volume, buy_vol, sell_vol
    """
    buf = defaultdict(lambda: {'open': None, 'high': -math.inf, 'low': math.inf, 'close': None,
                               'volume': 0.0, 'buy_vol': 0.0, 'sell_vol': 0.0})
    last_sec = None
    queue = deque(maxlen=lookback_seconds+5)

    async for tr in trades_stream(symbol):
        s = sec(tr['ts_ms'])
        b = buf[s]
        price = tr['price']
        qty = tr['qty']

        if b['open'] is None:
            b['open'] = price
        b['high'] = max(b['high'], price)
        b['low']  = min(b['low'], price)
        b['close'] = price
        b['volume'] += qty
        if tr['is_bm']:
            # If buyer is market maker (m=True), trade executed at bid => seller-initiated
            b['sell_vol'] += qty
        else:
            b['buy_vol'] += qty

        # finalize previous second when we move to a new one
        if last_sec is None:
            last_sec = s
        if s > last_sec:
            out = buf[last_sec]
            out_rec = {
                'ts': last_sec,
                **out
            }
            queue.append(out_rec)
            # cleanup to control memory
            keys_to_del = [k for k in buf.keys() if k < s - 2]
            for k in keys_to_del:
                buf.pop(k, None)
            last_sec = s
            yield out_rec
