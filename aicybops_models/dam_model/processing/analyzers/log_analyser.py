import logging
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

logger = logging.getLogger(__name__)


class LogAnalyzer:
    def __init__(self, log_file_path: Optional[str] = None, drain3_state_path: Optional[str] = None):
        self.logs = []

        self.log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)'  # Timestamp (ISO format)
            r'\s+(\w+)'                                        # Log level
            r'\s+(\d+)'                                        # Process ID
            r'\s+---\s+\[(.*?)\]'                             # Thread
            r'\s+(.*?)\s+:'                                   # Logger
            r'\s+(.*)'                                        # Message
        )

        if drain3_state_path is not None and drain3_state_path.strip():
            state_path = drain3_state_path.strip()
        else:
            state_path = os.environ.get("DAM_DRAIN3_STATE_PATH") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "drain3_state.bin"
            )
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)

        self.drain_config = TemplateMinerConfig()
        self.drain_config.mask_prefix = "<"    # Start marker for variable parts
        self.drain_config.mask_suffix = ">"    # End marker for variable parts
        self.drain_config.sim_th = 0.5         # Similarity threshold for clustering
        self.drain_config.max_depth = 7        # Maximum depth of the parse tree

        self.template_miner = self._init_template_miner(state_path)

        self.templates = {}
        self.template_counts = Counter()
        self.template_numbers = {}
        self.aligned_logs = {}
        self._aligned_log_df: Optional[pd.DataFrame] = None

        if log_file_path:
            self.load_log_file(log_file_path)
            self.extract_templates()
            self.assign_template_numbers()
            self._aligned_log_df = self.align_logs(align_freq='1s')

    def _init_template_miner(self, state_path: str) -> TemplateMiner:
        """Create a TemplateMiner, handling corrupt persisted state gracefully."""
        persistence = FilePersistence(state_path)
        try:
            return TemplateMiner(persistence_handler=persistence, config=self.drain_config)
        except RuntimeError:
            pass

        try:
            os.remove(state_path)
        except OSError:
            pass
        persistence = FilePersistence(state_path)
        try:
            return TemplateMiner(persistence_handler=persistence, config=self.drain_config)
        except RuntimeError:
            pass

        return TemplateMiner(persistence_handler=None, config=self.drain_config)

    def parse_log(self, log_line: str) -> Optional[Dict]:
        match = self.log_pattern.match(log_line)
        if match:
            timestamp, log_level, process_id, thread, logger_name, message = match.groups()
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return {
                'timestamp': timestamp,
                'datetime': dt,
                'log_level': log_level,
                'process_id': process_id,
                'thread': thread,
                'logger': logger_name,
                'message': message,
                'raw_message': log_line
            }
        return None

    def load_log_file(self, log_file_path: str) -> None:
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"Log file not found: {log_file_path}")

        with open(log_file_path, 'r') as file:
            for line in file:
                parsed_log = self.parse_log(line.strip())
                if parsed_log:
                    self.logs.append(parsed_log)

        logger.info("Loaded %d log entries", len(self.logs))

    def load_log_directory(self, directory_path: str, pattern: str = "LOGS_*.txt") -> None:
        import glob

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"The directory {directory_path} does not exist.")

        log_files = glob.glob(os.path.join(directory_path, pattern))

        for log_file in log_files:
            if "_structured." in log_file or "_templates." in log_file:
                continue
            logger.info("Processing log file: %s", log_file)
            self.load_log_file(log_file)

    def extract_templates(self) -> None:
        for log in self.logs:
            message = log.get('message', log.get('raw_message', ''))
            if message:
                result = self.template_miner.add_log_message(message)
                template_id = result['cluster_id']
                self.templates[template_id] = result['template_mined']
                self.template_counts[template_id] += 1
                log['template_id'] = template_id

    def assign_template_numbers(self) -> None:
        sorted_templates = sorted(
            self.templates.items(),
            key=lambda x: self.template_counts[x[0]]
        )

        total_templates = len(sorted_templates)
        for i, (template_id, _) in enumerate(sorted_templates):
            self.template_numbers[template_id] = total_templates - i

        for log in self.logs:
            if 'template_id' in log:
                log['template_number'] = self.template_numbers[log['template_id']]

    def align_logs(self, align_freq: str = '1s') -> pd.DataFrame:
        """
        Create time windows for the log data, grouping logs by time intervals.
        Each window contains a list of template numbers that occurred in that window.
        Returns DataFrame indexed by timestamp with log_template column.
        """
        df = pd.DataFrame([log for log in self.logs if 'template_number' in log])

        if df.empty:
            logger.warning("No logs with template numbers found")
            self.aligned_logs = {}
            return pd.DataFrame(columns=['log_template']).set_index(pd.DatetimeIndex([], name='timestamp'))

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['window'] = df['datetime'].dt.floor(align_freq)

        self.aligned_logs = df.groupby('window')['template_number'].apply(list).to_dict()

        window_sizes = [len(templates) for templates in self.aligned_logs.values()]
        logger.info("Time windows: %d windows, avg %.2f templates/window, min %d, max %d",
            len(self.aligned_logs), np.mean(window_sizes), min(window_sizes), max(window_sizes))

        log_df = pd.DataFrame([
            {"timestamp": ts, "log_template": max(templates) if templates else 0}
            for ts, templates in self.aligned_logs.items()
        ]).set_index("timestamp").sort_index()

        self._aligned_log_df = log_df
        return log_df

    def prepare_lstm_input(self, sequence_length: int = 10, max_templates_per_window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        sorted_windows = sorted(self.aligned_logs.items())

        timestamps = []
        sequences = []

        for i in range(len(sorted_windows) - sequence_length + 1):
            window_slice = sorted_windows[i:i + sequence_length]
            sequence = []
            for _, template_numbers in window_slice:
                window_templates = template_numbers[:max_templates_per_window]
                window_templates.extend([0] * (max_templates_per_window - len(window_templates)))
                sequence.append(window_templates)
            sequences.append(sequence)
            timestamps.append(window_slice[-1][0])

        return np.array(sequences, dtype=np.int32), np.array(timestamps)

    def get_template_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_templates": len(self.templates),
            "template_frequencies": dict(self.template_counts),
            "template_numbers": self.template_numbers,
        }

        most_common = [
            {
                "template_id": template_id,
                "template": self.templates[template_id],
                "frequency": count,
                "number": self.template_numbers[template_id]
            }
            for template_id, count in self.template_counts.most_common(10)
        ]
        stats["most_common_templates"] = most_common

        df = pd.DataFrame(self.logs)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            template_by_hour = df.groupby(['hour', 'template_id']).size().unstack(fill_value=0)
            stats["template_distribution_by_hour"] = template_by_hour.to_dict()

        return stats

    def export_processed_data(self, output_file: str) -> None:
        export_data = {
            "templates": {
                str(template_id): {
                    "text": template_text,
                    "frequency": self.template_counts[template_id],
                    "number": self.template_numbers[template_id]
                }
                for template_id, template_text in self.templates.items()
            },
            "time_windows": {
                str(timestamp): template_numbers
                for timestamp, template_numbers in self.aligned_logs.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info("Processed data exported to %s", output_file)

    def visualize_template_distribution(self, output_file: Optional[str] = None) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        template_frequencies = pd.Series(self.template_counts)
        template_frequencies.sort_values(ascending=False).plot(
            kind='bar',
            ax=ax1,
            title='Template Frequency Distribution'
        )
        ax1.set_xlabel('Template ID')
        ax1.set_ylabel('Frequency')
        ax1.tick_params(axis='x', rotation=45)

        df = pd.DataFrame(self.logs)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour

        pivot_table = df.pivot_table(
            index='hour',
            columns='template_id',
            values='template_number',
            aggfunc='count',
            fill_value=0
        )

        im = ax2.imshow(pivot_table, aspect='auto', cmap='YlOrRd')
        ax2.set_title('Template Occurrence by Hour')
        ax2.set_xlabel('Template ID')
        ax2.set_ylabel('Hour of Day')
        plt.colorbar(im, ax=ax2, label='Number of Occurrences')

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()

    def summarize_analysis(self) -> Dict[str, Any]:
        template_stats = self.get_template_statistics()

        if self.logs:
            start_time = min(log['datetime'] for log in self.logs)
            end_time = max(log['datetime'] for log in self.logs)
            time_range = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            }
        else:
            time_range = {}

        return {
            "overview": {
                "total_logs": len(self.logs),
                "total_templates": len(self.templates),
                "time_range": time_range
            },
            "template_analysis": template_stats
        }

    def get_processed_data(self, align_freq: str = '1s') -> pd.DataFrame:
        """
        Runs the log processing pipeline and returns a DataFrame indexed by time window,
        with a feature such as the max template number per window.
        align_freq: resampling/aggregation frequency (default '1s')
        """
        if self._aligned_log_df is not None:
            return self._aligned_log_df
        self.extract_templates()
        self.assign_template_numbers()
        return self.align_logs(align_freq=align_freq)

    def create_sequences(self, window_size: int = 10, stride: int = 1, align_freq: str = '1s'):
        """
        Create sliding windows (sequences) from the aligned log feature DataFrame (e.g., log_template).
        Each window contains window_size consecutive timesteps (at align_freq intervals), with stride between windows.
        Returns a numpy array of shape (n_windows, window_size, n_features).
        """
        log_df = self.get_processed_data(align_freq=align_freq)
        data = log_df.values
        seqs = []
        for i in range(0, len(data) - window_size + 1, stride):
            seqs.append(data[i:i+window_size])
        return np.array(seqs)
