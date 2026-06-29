import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone

_CHUNK_SIZE_MINUTES = 10             # size of each chunk (larger chunks to reduce request overhead)
_CHUNK_THRESHOLD_SECONDS = _CHUNK_SIZE_MINUTES * 60   # windows > 5 min trigger chunking (test-friendly)
_MAX_RETRIES = 3                     # per-chunk retry attempts

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

from .call_api import CallAPI, NonRetryableAPIError
from .session_log_to_labels import session_log_to_labels, report_unlabellable_periods


class DataCollector:
    """
    Collects data from API and saves it to files for use by DAMDataProcessor.
    """
    
    def __init__(self, api_url: Optional[str] = None, output_dir: str = None,
                 redis_url: Optional[str] = None):
        """
        Initialize DataCollector.
        
        Args:
            api_url: API base URL (from env var API_URL if not provided)
            output_dir: Directory to store collected data (required)
            redis_url: Optional Redis URL for direct session log reading
                       (bypasses API, gives real-time data during live runs)
        """
        self.api_url = api_url or os.getenv('API_URL')
        if not self.api_url:
            raise ValueError("API_URL must be provided either as parameter or environment variable")
        
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.call_api = CallAPI(self.api_url)
        self._redis_url = redis_url
        self._redis_client = None
        if redis_url and redis_lib is not None:
            try:
                self._redis_client = redis_lib.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
            except Exception as e:
                print(f"[DataCollector] Redis unavailable ({redis_url}): {e} — will use API")
                self._redis_client = None

    def test_connection(self, timeout: int = 10) -> Optional[dict]:
        """
        Call GET /test_connection to verify API reachability and optional InfluxDB connectivity.
        Returns the JSON response dict, or None on failure.
        """
        return self.call_api.test_connection(timeout=timeout)
    
    def _fetch_metrics_from_api(self, start: str = "-120s", stop: str | None = None) -> pd.DataFrame:
        """
        Fetch metrics from API.
        
        Returns:
            DataFrame with metrics data
        """
        try:
            print(f"[DataCollector] Fetching metrics from API...")
            # call_api appends 's' to start; pass e.g. "-120" so request gets "-120s"
            metrics_start = start.rstrip("s") if start.endswith("s") else start
            metrics_df = self.call_api.all_container_metrics(start=metrics_start, stop=stop, save_to_disk=False)
            if metrics_df is None:
                raise ValueError("Failed to fetch metrics from API")
            print(f"[DataCollector] Fetched {len(metrics_df)} metric rows")
            return metrics_df
        except Exception as e:
            print(f"[DataCollector] Error fetching metrics: {e}")
            raise
    
    def _fetch_logs_from_api(self, start: str = "-120s", stop: str | None = None, loganalyzer_format: bool = True):
        """
        Fetch logs from API via CallAPI.all_logs().

        When loganalyzer_format is True (default), requests format=loganalyzer and
        returns the response body as str (one line per log entry) for direct use in
        logs.txt. Otherwise returns JSON (unchanged behavior).
        Uses the same start/stop time range as metrics for alignment (e.g. session range).

        Returns:
            str when loganalyzer_format=True (LogAnalyzer-compatible text), else JSON.
        """
        try:
            print("[DataCollector] Fetching logs from API...")
            fmt = "loganalyzer" if loganalyzer_format else None
            logs_data = self.call_api.all_logs(start=start, stop=stop, format=fmt)
            if logs_data is None:
                raise ValueError("Failed to fetch logs from API")
            print("[DataCollector] Fetched logs from API")
            return logs_data
        except Exception as e:
            print(f"[DataCollector] Error fetching logs: {e}")
            raise
    
    def _convert_logs_json_to_text(self, logs_json: Any) -> List[str]:
        """
        Convert JSON logs from API to text format compatible with LogAnalyzer.
        
        LogAnalyzer expects format: {timestamp} {level} {pid} --- [{thread}] {logger} : {message}
        Example: 2025-01-01T00:01:00.002Z INFO 1006 --- [io-1010-exec-10] logger : message
        
        Args:
            logs_json: JSON logs data (may be list of dicts, single dict, or other structure)
        
        Returns:
            List of formatted log strings (one per line)
        """
        formatted_logs = []
        
        if logs_json is None:
            return formatted_logs
        
        # Handle different JSON structures
        if isinstance(logs_json, list):
            log_entries = logs_json
        elif isinstance(logs_json, dict):
            # If it's a dict, try to extract list of logs
            if 'logs' in logs_json:
                log_entries = logs_json['logs']
            elif 'data' in logs_json:
                log_entries = logs_json['data']
            else:
                # Single log entry as dict
                log_entries = [logs_json]
        else:
            # Try to convert to list
            log_entries = [logs_json]
        
        for log_entry in log_entries:
            if not isinstance(log_entry, dict):
                continue
            
            # Extract fields with defaults
            timestamp = log_entry.get('timestamp') or log_entry.get('time') or log_entry.get('_time', '')
            log_level = log_entry.get('level') or log_entry.get('log_level') or log_entry.get('severity', 'INFO')
            process_id = log_entry.get('pid') or log_entry.get('process_id') or log_entry.get('processId', '0')
            thread = log_entry.get('thread') or log_entry.get('thread_name') or log_entry.get('threadName', '')
            logger = log_entry.get('logger') or log_entry.get('logger_name') or log_entry.get('loggerName', 'logger')
            message = log_entry.get('message') or log_entry.get('msg') or log_entry.get('text', '')
            
            # Format timestamp to ISO format if needed
            if timestamp:
                try:
                    # Try to parse and reformat if needed
                    if isinstance(timestamp, (int, float)):
                        # Unix timestamp
                        dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
                        timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    elif isinstance(timestamp, str):
                        # Try to parse and ensure ISO format
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        except:
                            # Keep as is if parsing fails
                            pass
                except:
                    # Keep original if conversion fails
                    pass
            
            # Format as: {timestamp} {level} {pid} --- [{thread}] {logger} : {message}
            formatted_log = f"{timestamp} {log_level} {process_id} --- [{thread}] {logger} : {message}"
            formatted_logs.append(formatted_log)
        
        print(f"[DataCollector] Converted {len(formatted_logs)} log entries to text format")
        return formatted_logs
    
    def _normalise_metrics_for_dam(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert API long format (_time, _measurement, _field, _value) to DAM CSV schema
        (_time, _measurement, counter, gauge, operation) so MetricsAnalyser finds _measurement
        and value columns.
        """
        if metrics_df.empty:
            return metrics_df
        # Already has DAM schema (e.g. from file)
        if "_measurement" in metrics_df.columns and ("counter" in metrics_df.columns or "gauge" in metrics_df.columns):
            return metrics_df
        # API long format: add counter/gauge from _value so analyser can use value_column from config
        if "_measurement" in metrics_df.columns and "_value" in metrics_df.columns:
            if "counter" not in metrics_df.columns:
                metrics_df["counter"] = metrics_df["_value"]
            if "gauge" not in metrics_df.columns:
                metrics_df["gauge"] = metrics_df["_value"]
            if "operation" not in metrics_df.columns:
                metrics_df["operation"] = ""
        # Ensure _time is Unix milliseconds for MetricsAnalyser (pd.to_datetime(..., unit='ms')).
        # Use the actual point timestamp (_time), never _start (query range start, same for all rows).
        # API/InfluxDB may send _time as ms (~1e12), ns (>1e15), or datetime64.
        if "_time" in metrics_df.columns:
            raw = pd.to_numeric(metrics_df["_time"], errors="coerce")
            if raw.notna().any():
                rmax = raw.max()
                if rmax > 1e15:
                    metrics_df["_time"] = (raw // 10**6).astype("int64")  # ns -> ms
                elif 1e12 <= rmax <= 1e14:
                    metrics_df["_time"] = raw.astype("int64")  # already ms
                elif rmax <= 1e9:
                    metrics_df["_time"] = (raw * 1000).astype("int64")  # s -> ms
                else:
                    metrics_df["_time"] = raw.astype("int64")
            else:
                ts = pd.to_datetime(metrics_df["_time"])
                metrics_df["_time"] = (ts.astype("int64") // 10**6).astype("int64")
        return metrics_df

    def _save_collected_data(self, metrics_df: pd.DataFrame, logs_data: Any) -> Dict[str, str]:
        """
        Save collected data to files.

        logs_data may be:
        - str: LogAnalyzer-compatible text (from API with format=loganalyzer), written as-is.
        - JSON (list/dict): converted via _convert_logs_json_to_text then written.

        Args:
            metrics_df: DataFrame with metrics data
            logs_data: Logs as str (loganalyzer text) or JSON

        Returns:
            Dictionary with paths to saved files
        """
        metrics_df = self._normalise_metrics_for_dam(metrics_df)
        # Save metrics to CSV
        metrics_path = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[DataCollector] Saved metrics to {metrics_path}")

        log_path = self.output_dir / "logs.txt"
        if isinstance(logs_data, str):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(logs_data)
        else:
            log_text_lines = self._convert_logs_json_to_text(logs_data)
            with open(log_path, "w", encoding="utf-8") as f:
                for line in log_text_lines:
                    f.write(line + "\n")
        print(f"[DataCollector] Saved logs to {log_path}")
        
        return {
            'metrics_csv_path': str(metrics_path),
            'log_file_path': str(log_path)
        }

    def _count_collected(self, metrics_df: pd.DataFrame, logs_data: Any) -> Tuple[int, int]:
        """
        Count the number of metrics (rows) and log entries collected from the API.

        Args:
            metrics_df: DataFrame with metrics data (from _fetch_metrics_from_api).
            logs_data: Logs as str (loganalyzer text, one line per entry) or JSON.

        Returns:
            (n_metrics, n_logs)
        """
        n_metrics = len(metrics_df) if metrics_df is not None else 0
        n_logs = 0
        if isinstance(logs_data, str):
            n_logs = sum(1 for line in logs_data.splitlines() if line.strip())
        elif isinstance(logs_data, list):
            n_logs = len(logs_data)
        elif isinstance(logs_data, dict):
            if "logs" in logs_data:
                n_logs = len(logs_data["logs"]) if isinstance(logs_data["logs"], list) else 0
            elif "data" in logs_data:
                n_logs = len(logs_data["data"]) if isinstance(logs_data["data"], list) else 0
            else:
                n_logs = 1
        return n_metrics, n_logs

    @staticmethod
    def _parse_iso_utc(ts: str) -> Optional[datetime]:
        """Parse an ISO timestamp string to a timezone-aware UTC datetime."""
        try:
            s = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None

    def _session_time_range(
        self, session_log: Dict[str, Any], buffer_seconds: int = 120,
        training_window_minutes: int = 0,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Compute Flux-compatible start/stop from session_log timestamps.
        When session_end exists, returns an absolute ISO stop so only
        the actual session window is queried (avoids huge queries when
        the test runs long after the session ended).

        When training_window_minutes > 0, caps stop at session_start + N minutes
        (or now, whichever is earlier). This allows training on a subset of the session.

        Returns (start, stop) — stop is None when session is still in progress
        and no training_window_minutes is set.
        """
        start_dt = self._parse_iso_utc(session_log.get("session_start", ""))
        if start_dt is None:
            return None, None
        now = datetime.now(timezone.utc)
        delta = (now - start_dt).total_seconds()
        if delta < 0:
            return None, None
        start = f"-{int(delta) + buffer_seconds}s"

        # Determine stop time
        stop = None
        if training_window_minutes > 0:
            # Cap to first N minutes of session (or now, whichever is earlier)
            window_end = start_dt + timedelta(minutes=training_window_minutes)
            stop_dt = min(window_end, now)
            stop = (stop_dt + timedelta(seconds=buffer_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_ts = session_log.get("session_end")
            if end_ts:
                end_dt = self._parse_iso_utc(end_ts)
                if end_dt is not None:
                    stop = (end_dt + timedelta(seconds=buffer_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return start, stop

    def _get_session_log_from_redis(self, key: str = "session_log") -> Optional[Dict[str, Any]]:
        """Read session log directly from Redis. Returns dict or None."""
        if self._redis_client is None:
            return None
        try:
            raw = self._redis_client.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and "normal_windows" in data:
                return data
            return None
        except Exception as e:
            print(f"[DataCollector] Redis session_log read failed: {e}")
            return None

    def _get_session_log(self) -> Optional[Dict[str, Any]]:
        """Get session log: Redis first (real-time), then API (Redis-backed) fallback."""
        if self._redis_client is not None:
            log = self._get_session_log_from_redis()
            if log is not None:
                print("[DataCollector] Session log read from Redis (real-time)")
                return log
        return self.call_api.get_session_log()

    def _fetch_session_log_labels(
        self,
        labels_path: Path,
        metrics_df: Optional[pd.DataFrame] = None,
        session_log: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Convert session log to anomaly_labels.csv using metrics timestamps (1:1).
        If session_log is None, fetches it from the API. metrics_df is required for label generation.
        Returns the labels file path on success, None if session log unavailable or generation fails.
        """
        if session_log is None:
            print("[DataCollector] Fetching session log from API for labels...")
            session_log = self.call_api.get_session_log()
        if not session_log:
            print("[DataCollector] No session log available — labels will not be generated.")
            return None
        if metrics_df is None or metrics_df.empty:
            print("[DataCollector] No metrics_df provided — labels skipped (need 1:1 with metrics).")
            return None
        result = session_log_to_labels(session_log, str(labels_path), metrics_df=metrics_df)
        if result:
            print(f"[DataCollector] Labels generated from session log -> {labels_path}")
            periods = report_unlabellable_periods(str(labels_path))
            if periods:
                print("[DataCollector] Unlabellable periods (anomaly_label=-1):")
                for start_ts, end_ts in periods:
                    print(f"  {start_ts} to {end_ts}")
        else:
            print("[DataCollector] session_log_to_labels failed (missing session_start/end?) — labels skipped.")
        return result

    # ------------------------------------------------------------------
    # Chunked collection helpers
    # ------------------------------------------------------------------

    def _parse_start_to_dt(self, start: str, now: datetime) -> Optional[datetime]:
        """Parse a start value (e.g. '-3600s' or ISO string) to a UTC datetime."""
        s = start.strip()
        if s.lstrip("-").rstrip("s").isdigit() and s.endswith("s"):
            offset = int(s.rstrip("s"))  # negative integer
            return now + timedelta(seconds=offset)
        return self._parse_iso_utc(s)

    def _should_chunk(self, start: str, stop: Optional[str]) -> bool:
        """Return True if the time window exceeds _CHUNK_THRESHOLD_SECONDS."""
        now = datetime.now(timezone.utc)
        start_dt = self._parse_start_to_dt(start, now)
        stop_dt = self._parse_iso_utc(stop) if stop else now
        if start_dt is None or stop_dt is None:
            return False
        return (stop_dt - start_dt).total_seconds() > _CHUNK_THRESHOLD_SECONDS

    def _generate_chunk_ranges(self, start: str, stop: Optional[str]) -> List[Tuple[str, str]]:
        """Split the window into _CHUNK_SIZE_MINUTES-sized slices of (start_iso, stop_iso)."""
        now = datetime.now(timezone.utc)
        start_dt = self._parse_start_to_dt(start, now)
        stop_dt = self._parse_iso_utc(stop) if stop else now
        if start_dt is None or stop_dt is None:
            return []
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        chunk_delta = timedelta(minutes=_CHUNK_SIZE_MINUTES)
        chunks = []
        current = start_dt
        while current < stop_dt:
            chunk_end = min(current + chunk_delta, stop_dt)
            chunks.append((current.strftime(fmt), chunk_end.strftime(fmt)))
            current = chunk_end
        return chunks

    def _fetch_chunk_with_retry(self, chunk_start: str, chunk_stop: str) -> Tuple[pd.DataFrame, Any, int]:
        """Fetch one (chunk_start, chunk_stop) window with exponential backoff retry."""
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                metrics_df = self._fetch_metrics_from_api(start=chunk_start, stop=chunk_stop)
                logs_data = self._fetch_logs_from_api(start=chunk_start, stop=chunk_stop)
                return metrics_df, logs_data, attempt + 1
            except NonRetryableAPIError as exc:
                raise RuntimeError(
                    f"[DataCollector] Chunk {chunk_start}→{chunk_stop} failed with non-retryable error: {exc}"
                ) from exc
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"[DataCollector] Chunk {chunk_start}→{chunk_stop} attempt {attempt + 1} failed: {exc}. "
                          f"Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
        raise RuntimeError(
            f"[DataCollector] Chunk {chunk_start}→{chunk_stop} failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _get_data_chunked(
        self, start: str, stop: Optional[str], session_log: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Collect metrics and logs in time-range chunks, appending each to disk."""
        chunks = self._generate_chunk_ranges(start, stop)
        print(f"[DataCollector] Chunked collection: {len(chunks)} chunk(s) of {_CHUNK_SIZE_MINUTES} min", flush=True)

        metrics_path = self.output_dir / "metrics.csv"
        log_path = self.output_dir / "logs.txt"
        total_metrics = 0
        total_logs = 0

        for i, (chunk_start, chunk_stop) in enumerate(chunks):
            chunk_idx = i + 1
            chunk_t0 = time.monotonic()
            print(
                f"[DataCollector] chunk_started chunk={chunk_idx}/{len(chunks)} start={chunk_start} stop={chunk_stop}",
                flush=True,
            )
            try:
                metrics_df, logs_data, attempts_used = self._fetch_chunk_with_retry(chunk_start, chunk_stop)
                metrics_df = self._normalise_metrics_for_dam(metrics_df)

                # metrics: header on first chunk, append-only on subsequent
                metrics_df.to_csv(
                    metrics_path,
                    mode="w" if i == 0 else "a",
                    header=(i == 0),
                    index=False,
                )

                # logs: overwrite on first chunk, append on subsequent
                if isinstance(logs_data, str):
                    log_text = logs_data
                else:
                    log_text = "\n".join(self._convert_logs_json_to_text(logs_data))
                with open(log_path, "w" if i == 0 else "a", encoding="utf-8") as f:
                    f.write(log_text if i == 0 else "\n" + log_text)

                n_m, n_l = self._count_collected(metrics_df, logs_data)
                total_metrics += n_m
                total_logs += n_l
                duration_ms = int((time.monotonic() - chunk_t0) * 1000)
                print(
                    f"[DataCollector] chunk_completed chunk={chunk_idx}/{len(chunks)} metrics_rows={n_m} "
                    f"logs_rows={n_l} retries={max(0, attempts_used - 1)} duration_ms={duration_ms}",
                    flush=True,
                )
            except Exception as exc:
                duration_ms = int((time.monotonic() - chunk_t0) * 1000)
                print(
                    f"[DataCollector] chunk_failed chunk={chunk_idx}/{len(chunks)} start={chunk_start} stop={chunk_stop} "
                    f"duration_ms={duration_ms} error_type={type(exc).__name__} error={exc}",
                    flush=True,
                )
                raise

        print(f"[DataCollector] Chunked collection complete: {total_metrics} metrics, {total_logs} logs", flush=True)

        # label generation: read back only _time column to avoid reloading full CSV
        full_metrics_times = pd.read_csv(metrics_path, usecols=["_time"])
        labels_path = self.output_dir / "anomaly_labels.csv"
        labels_result = self._fetch_session_log_labels(
            labels_path, metrics_df=full_metrics_times, session_log=session_log
        )

        paths: Dict[str, str] = {
            "metrics_csv_path": str(metrics_path),
            "log_file_path": str(log_path),
            "metrics_count": str(total_metrics),
            "logs_count": str(total_logs),
        }
        if labels_result:
            paths["labels_csv_path"] = labels_result
        return paths

    def get_data(
        self,
        start: str = "-120s",
        use_session_time_range: bool = True,
        training_window_minutes: int = 0,
    ) -> Dict[str, str]:
        """
        Fetch metrics, logs, and labels (from session log) from API and save to files.
        When a session log is available and use_session_time_range is True, metrics and
        logs are fetched for the session time range so labels (1:1 with metrics) align.
        Otherwise uses start (e.g. -120s). Set use_session_time_range=False to force the
        given start (e.g. for testing when session range predates available data).

        Args:
            training_window_minutes: If > 0, limits data to the first N minutes of the
                session (from session_start to min(session_start + N min, now)).

        Returns:
            Dictionary with keys 'metrics_csv_path', 'log_file_path', and optionally
            'labels_csv_path' (only when a session log was available).
        """
        try:
            session_log = self._get_session_log()
            stop = None
            if session_log and use_session_time_range:
                sess_start, sess_stop = self._session_time_range(
                    session_log, training_window_minutes=training_window_minutes,
                )
                if sess_start is not None:
                    start = sess_start
                    stop = sess_stop
                    window_note = f" (first {training_window_minutes} min)" if training_window_minutes > 0 else ""
                    print(f"[DataCollector] Using session time range: start={start}, stop={stop or 'now'}{window_note}")
            # else: keep caller's start when use_session_time_range=False

            # Use chunked collection for large windows (> 30 min) to avoid OOM
            if self._should_chunk(start, stop):
                return self._get_data_chunked(start, stop, session_log)

            metrics_df = self._fetch_metrics_from_api(start=start, stop=stop)
            if metrics_df is None or metrics_df.empty:
                raise ValueError(
                    "API returned no metrics. Check API_URL, collect_metrics API, and that InfluxDB has data for the requested range."
                )
            if "_measurement" not in metrics_df.columns:
                raise ValueError(
                    "API response missing '_measurement' column (required by DAM). "
                    "Ensure collect_metrics API returns long-format data with _time, _measurement, _field, _value."
                )
            logs_data = self._fetch_logs_from_api(start=start, stop=stop)

            n_metrics, n_logs = self._count_collected(metrics_df, logs_data)
            print(f"[DataCollector] Collected: {n_metrics} metrics, {n_logs} logs")

            paths = self._save_collected_data(metrics_df, logs_data)
            paths["metrics_count"] = str(n_metrics)
            paths["logs_count"] = str(n_logs)

            labels_path = self.output_dir / "anomaly_labels.csv"
            labels_result = self._fetch_session_log_labels(
                labels_path, metrics_df=metrics_df, session_log=session_log
            )
            if labels_result:
                paths["labels_csv_path"] = labels_result

            return paths
        except Exception as e:
            print(f"[DataCollector] Error in get_data(): {e}")
            raise

