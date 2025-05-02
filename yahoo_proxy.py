import requests
import json as _json
import io
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Tuple

# --- Конфигурация Прокси ---
CLOUD_FUNCTION_URL = "https://us-central1-gen-lang-client-0158296627.cloudfunctions.net/function-0"
YAHOO_DOMAINS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")

# --- Пользовательский класс Session для проксирования запросов к Yahoo --- 
class ProxiedYahooSession(requests.Session):
    def request(self, method, url, params=None, data=None, headers=None, cookies=None, files=None, auth=None, timeout=None, allow_redirects=True, proxies=None, hooks=None, stream=None, verify=None, cert=None, json=None):
        
        is_yahoo_request = any(domain in url for domain in YAHOO_DOMAINS)

        if is_yahoo_request:
            print(f"Intercepting Yahoo request: {method} {url}")
            proxied_headers = dict(headers) if headers else {}
            target_url_with_params = url
            if params:
                 query_string = requests.compat.urlencode(params)
                 target_url_with_params += "?" + query_string

            payload = {
                "url": target_url_with_params,
                "method": method,
                "headers": proxied_headers
            }

            try:
                proxy_response = requests.post(CLOUD_FUNCTION_URL, json=payload, timeout=timeout or 30)
                print(f"  Proxy Function status: {proxy_response.status_code}")

                final_response = requests.Response()
                final_response.elapsed = proxy_response.elapsed
                final_response.request = proxy_response.request
                final_response.url = url
                
                if not proxy_response.ok:
                    print(f"  Error: Proxy function returned status {proxy_response.status_code}")
                    final_response.status_code = proxy_response.status_code
                    final_response._content = proxy_response.content
                    final_response.headers = proxy_response.headers
                    return final_response

                try:
                    proxy_json = proxy_response.json()
                    if 'data' not in proxy_json:
                        print(f"  Error: Key 'data' not found in proxy response.")
                        final_response.status_code = 502
                        final_response._content = b'{"error": "Proxy did not return data key"}'
                        final_response.headers = {'Content-Type': 'application/json'}
                        return final_response
                    
                    actual_yahoo_data = proxy_json['data']
                    actual_yahoo_content_bytes = _json.dumps(actual_yahoo_data).encode('utf-8')
                    yahoo_error = actual_yahoo_data.get('chart', {}).get('error')
                    
                    final_response.status_code = 404 if yahoo_error else 200
                    final_response._content = actual_yahoo_content_bytes
                    final_response.encoding = 'utf-8'
                    final_response.headers = {'Content-Type': 'application/json'} 
                    print(f"  Success: Returning proxied data for {url} with fabricated status {final_response.status_code}")
                    return final_response
                
                except _json.JSONDecodeError:
                    print("  Error: Failed to decode JSON from proxy response.")
                    final_response.status_code = 502
                    final_response._content = b'{"error": "Proxy response not valid JSON"}'
                    final_response.headers = {'Content-Type': 'application/json'}
                    return final_response

            except requests.exceptions.RequestException as e:
                print(f"  Error connecting to proxy function: {e}")
                final_response = requests.Response()
                final_response.status_code = 504
                final_response.url = url
                final_response._content = b'{"error": "Failed to connect to proxy"}'
                final_response.headers = {'Content-Type': 'application/json'}
                return final_response
        else:
            return super().request(method, url, params=params, data=data, headers=headers, cookies=cookies, files=files, auth=auth, timeout=timeout, allow_redirects=allow_redirects, proxies=proxies, hooks=hooks, stream=stream, verify=verify, cert=cert, json=json)

# --- Функция-обертка для загрузки через прокси ---
def download_via_proxy(tickers: List[str], start: Optional[str] = None, end: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
    """
    Вызывает yfinance.download с auto_adjust=False через прокси-сессию.
    Принимает стандартные аргументы yfinance.download.
    """
    print("Using download_via_proxy function (expecting Adj Close)...")
    # Устанавливаем параметры yfinance для получения 'Adj Close'
    kwargs.setdefault('auto_adjust', False)
    kwargs.setdefault('repair', False)
    kwargs.setdefault('keepna', True)
    kwargs.setdefault('progress', False)
    kwargs.pop('session', None)

    result = None
    with ProxiedYahooSession() as session:
        try:
            result = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                session=session,
                **kwargs
            )
        except Exception as e:
            print(f"Error during download_via_proxy (calling yf.download): {e}")
            import traceback
            traceback.print_exc()
            return None

    if result is not None and result.empty:
         print(f"Warning: download_via_proxy returned empty DataFrame for {tickers}.")
         # Возвращаем пустой DataFrame
         return pd.DataFrame()

    return result 