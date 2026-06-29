import json
import os
import time

import requests
import pandas as pd

CONNECT_TIMEOUT_SECONDS = 3
READ_TIMEOUT_SECONDS = 600
LOGIN_READ_TIMEOUT_SECONDS = 10


class NonRetryableAPIError(Exception):
    """Raised for non-retryable HTTP conditions (most 4xx responses)."""


class CallAPI:

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    @staticmethod
    def _is_non_retryable_status(status_code: int) -> bool:
        return 400 <= status_code < 500 and status_code != 429

    def test_connection(self, timeout: int = 10) -> dict | None:
        """GET /test_connection to verify API reachability and optional InfluxDB connectivity."""
        url = f"{self.base_url}/test_connection"
        try:
            response = self.session.get(url, timeout=(CONNECT_TIMEOUT_SECONDS, timeout))
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"test_connection failed: {e}", flush=True)
            return None

    def _login(self) -> str | None:
        login_endpoint = f"{self.base_url}/login"
        login_data = {
            'username': 'test_user_api',
            'password': 'test_password_api'
        }
        start_time = time.monotonic()
        response = self.session.post(
            login_endpoint,
            json=login_data,
            timeout=(CONNECT_TIMEOUT_SECONDS, LOGIN_READ_TIMEOUT_SECONDS),
        )
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        if response.status_code == 200:
            print(f"[CallAPI] login status=200 duration_ms={elapsed_ms}", flush=True)
            return response.json().get('access_token')
        if self._is_non_retryable_status(response.status_code):
            raise NonRetryableAPIError(f"Login failed with status {response.status_code}")
        print(f"[CallAPI] login failed status={response.status_code} duration_ms={elapsed_ms}", flush=True)
        
        return None

    def get_session_log(self, session_log_url: str | None = None, timeout: int = 10) -> dict | None:
        """
        GET session log (testbed fault_events, normal_windows) for remote label collection.
        Uses same auth as other endpoints. If session_log_url is None, uses {base_url}/session_log.
        """
        url = session_log_url or f"{self.base_url}/session_log"
        token = self._login()
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "gzip, deflate"}
        try:
            start_time = time.monotonic()
            response = self.session.get(
                url,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_SECONDS, timeout),
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            if self._is_non_retryable_status(response.status_code):
                raise NonRetryableAPIError(f"Session log request failed with status {response.status_code}")
            response.raise_for_status()
            print(f"[CallAPI] session_log status={response.status_code} duration_ms={elapsed_ms}", flush=True)
            return response.json()
        except requests.RequestException as e:
            print(f"get_session_log failed: {e}", flush=True)
            return None

    def _save_to_disk(self, df, directory, file_name):
        if os.path.exists(directory):
            print(f"Directory '{directory}' already exists.")
        else:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Directory '{directory}' created successfully.")
        df.to_csv(directory+"/"+file_name, index=False)


    def all_container_metrics(
        self,
        start: str = '-120',
        stop: str | None = None,
        save_to_disk: bool = False,
    ):
        """Fetch all container metrics from the API (no measurement filter; filtering is done in DAM pipeline from config)."""
        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_metrics"

        token = self._login()
        
        df_cpu = None

        if token:
            headers = {'Authorization': f'Bearer {token}', 'Accept-Encoding': 'gzip, deflate'}
            # Append 's' only for relative offsets (e.g. '-120' → '-120s').
            # ISO timestamps (e.g. '2026-04-07T11:19:48Z') are passed as-is.
            start_param = start if "T" in start else f'{start}s'
            params = {'start': start_param}
            if stop is not None:
                params['stop'] = stop
            request_start = time.monotonic()
            response = self.session.get(
                cpu_endpoint,
                headers=headers,
                params=params,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                )
            elapsed_ms = int((time.monotonic() - request_start) * 1000)
            
            if response.status_code == 200:
                cpu_data = response.json()
                df_cpu = pd.DataFrame(cpu_data if isinstance(cpu_data, list) else [])
                # Ensure empty response still has _time column for downstream (MetricsAnalyser)
                if df_cpu.empty or "_time" not in df_cpu.columns:
                    df_cpu = pd.DataFrame(columns=["_time"])
                print(
                    f"[CallAPI] metrics status=200 start={start_param} stop={stop or 'now'} "
                    f"rows={len(df_cpu)} duration_ms={elapsed_ms}",
                    flush=True,
                )

                if save_to_disk:
                    directory = './main_tests_api/'
                    file_name = "main_test.csv"
                    self._save_to_disk(df_cpu, directory, file_name)


            else:
                if self._is_non_retryable_status(response.status_code):
                    raise NonRetryableAPIError(
                        f"Metrics request failed with non-retryable status {response.status_code}"
                    )
                print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

        
        return df_cpu

    def all_logs(
        self,
        start: str = "-2m",
        stop: str | None = None,
        format: str | None = None,
        timeout: int = 60,
    ):
        """
        GET /collect_logs/all_logs with Bearer token.

        - format='loganalyzer': response body is LogAnalyzer-compatible text (one line per
          log entry). Returns the raw response text (str) for writing directly to logs.txt.
        - format omitted or format != 'loganalyzer': response is JSON. Returns response.json().

        Args:
            start: Time range start, e.g. '-2m'.
            stop: Optional end time (e.g. ISO '2026-03-16T10:51:54Z'). When provided with
                  session range, keeps logs aligned with metrics.
            format: If 'loganalyzer', request format=loganalyzer and return response.text.
            timeout: Request timeout in seconds.

        Returns:
            str when format='loganalyzer', otherwise the parsed JSON (list/dict).
        """
        endpoint = f"{self.base_url}/collect_logs/all_logs"
        token = self._login()
        if not token:
            return None
        headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "gzip, deflate"}
        params = {"start": start}
        if stop is not None:
            params["stop"] = stop
        if format == "loganalyzer":
            params["format"] = "loganalyzer"
        request_start = time.monotonic()
        response = self.session.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=(CONNECT_TIMEOUT_SECONDS, timeout),
        )
        elapsed_ms = int((time.monotonic() - request_start) * 1000)
        if response.status_code != 200:
            if self._is_non_retryable_status(response.status_code):
                raise NonRetryableAPIError(
                    f"Logs request failed with non-retryable status {response.status_code}"
                )
            print(f"Failed to fetch logs: {response.status_code}", flush=True)
            return None
        print(
            f"[CallAPI] logs status=200 start={start} stop={stop or 'now'} format={format or 'json'} "
            f"duration_ms={elapsed_ms}",
            flush=True,
        )
        if format == "loganalyzer":
            return response.text
        return response.json()

    def all_container_metrics_start_stop(self, start, stop, aux):
        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_metrics_start_stop"
        response = requests.get(cpu_endpoint, params={'start': start, 'stop': stop})  

        if response.status_code == 200:

            print("DEU CERTO", flush=True)
            cpu_data = response.json()
            df_cpu = pd.DataFrame(cpu_data)
            print("CPU Metrics:", flush=True)
            print(df_cpu, flush=True)

        else:

            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)


        print("Also saving on folder: main_tests")
        directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/'
        if os.path.exists(directory):
            print(f"Directory '{directory}' already exists.")
        else:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Directory '{directory}' created successfully.")
        df_cpu.to_csv(directory+"/"+f"main_test{aux}.csv", index=False)

    def all_container_cpu_metrics(self):

        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_cpu_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-5m'})

        if response.status_code == 200:
            print("Request successful", flush=True)
            json_data_frames = response.json()

            # Convert JSON data frames to pandas DataFrames
            data_frames = []
            for df_json in json_data_frames:
                if isinstance(df_json, str):  
                    df_data = json.loads(df_json)  
                    df = pd.DataFrame(df_data)  
                    data_frames.append(df)
                elif isinstance(df_json, dict):  
                    df = pd.DataFrame(df_json)  
                    data_frames.append(df)
                else:
                    print("Invalid JSON data:", df_json)

            directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_cpu_metrics'
            os.makedirs(directory, exist_ok=True)  

            for i, df in enumerate(data_frames):
                # Generate the file name (e.g., df_1.csv, df_2.csv, ...)
                file_name = f'df_{i}.csv'  
                file_path = os.path.join(directory, file_name)

                # Save the DataFrame to CSV
                df.to_csv(file_path, index=False)  

                print(f"DataFrame {i+1} saved to '{file_path}'", flush=True)
        else:
            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

    def all_container_memory_metrics(base_url):
        cpu_endpoint = f"{base_url}/collect_metrics/all_container_memory_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-5m'})

        if response.status_code == 200:
            print("Request successful", flush=True)
            json_data_frames = response.json()

            # Convert JSON data frames to pandas DataFrames
            data_frames = []
            for df_json in json_data_frames:
                if isinstance(df_json, str):  
                    df_data = json.loads(df_json)  
                    df = pd.DataFrame(df_data)  
                    data_frames.append(df)
                elif isinstance(df_json, dict):  
                    df = pd.DataFrame(df_json)  
                    data_frames.append(df)
                else:
                    print("Invalid JSON data:", df_json)

            directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_memory_metrics'
            os.makedirs(directory, exist_ok=True)  

            for i, df in enumerate(data_frames):
                file_name = f'df_{i}.csv'
                file_path = os.path.join(directory, file_name)

                df.to_csv(file_path, index=False)

                print(f"DataFrame {i+1} saved to '{file_path}'", flush=True)
        else:
            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

    def all_container_fs_metrics(self):

        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_fs_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-5m'})

        if response.status_code == 200:
            print("Request successful", flush=True)
            json_data_frames = response.json()

            # Convert JSON data frames to pandas DataFrames
            data_frames = []
            for df_json in json_data_frames:
                if isinstance(df_json, str):  
                    df_data = json.loads(df_json) 
                    df = pd.DataFrame(df_data)  
                    data_frames.append(df)
                elif isinstance(df_json, dict):  
                    df = pd.DataFrame(df_json)  
                    data_frames.append(df)
                else:
                    print("Invalid JSON data:", df_json)

            directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_fs_metrics'
            os.makedirs(directory, exist_ok=True)  

            for i, df in enumerate(data_frames):
                
                file_name = f'df_{i}.csv'  
                file_path = os.path.join(directory, file_name)

                df.to_csv(file_path, index=False)  

                print(f"DataFrame {i+1} saved to '{file_path}'", flush=True)
        else:
            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

    def all_container_network_metrics(self):
        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_network_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-15m'})
        if response.status_code == 200:

            print("DEU CERTO", flush=True)
            cpu_data = response.json()
            df_cpu = pd.DataFrame(cpu_data)
            print("CPU Metrics:", flush=True)
            print(df_cpu, flush=True)

        else:

            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)


        print("Also saving on folder: main_tests")
        directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_network_metrics'
        if os.path.exists(directory):
            print(f"Directory '{directory}' already exists.")
        else:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Directory '{directory}' created successfully.")
        df_cpu.to_csv(directory+"/"+"main_test.csv", index=False)
        '''if response.status_code == 200:
            print("Request successful", flush=True)
            json_data_frames = response.json()

            # Convert JSON data frames to pandas DataFrames
            data_frames = []
            for df_json in json_data_frames:
                if isinstance(df_json, str):  
                    df_data = json.loads(df_json)  
                    df = pd.DataFrame(df_data)  
                    data_frames.append(df)
                elif isinstance(df_json, dict):  
                    df = pd.DataFrame(df_json)  
                    data_frames.append(df)
                else:
                    print("Invalid JSON data:", df_json)

            directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_network_metrics'
            os.makedirs(directory, exist_ok=True)  

            for i, df in enumerate(data_frames):
                # Generate the file name (e.g., df_1.csv, df_2.csv, ...)
                file_name = f'df_{i}.csv'  # Use i+1 to start numbering from 1
                file_path = os.path.join(directory, file_name)

                # Save the DataFrame to CSV
                df.to_csv(file_path, index=False)  # Set index=False to exclude row numbers in CSV

                print(f"DataFrame {i+1} saved to '{file_path}'", flush=True)
        else:
            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)'''

    def all_container_spec_metrics(self):
        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_spec_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-5m'})

        if response.status_code == 200:
            print("Request successful", flush=True)
            json_data_frames = response.json()

            # Convert JSON data frames to pandas DataFrames
            data_frames = []
            for df_json in json_data_frames:
                if isinstance(df_json, str):  
                    df_data = json.loads(df_json)  
                    df = pd.DataFrame(df_data)  
                    data_frames.append(df)
                elif isinstance(df_json, dict):  
                    df = pd.DataFrame(df_json)  
                    data_frames.append(df)
                else:
                    print("Invalid JSON data:", df_json)

            directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/all_container_spec_metrics'
            os.makedirs(directory, exist_ok=True)  

            for i, df in enumerate(data_frames):
                file_name = f'df_{i}.csv'  
                file_path = os.path.join(directory, file_name)

                df.to_csv(file_path, index=False)  

                print(f"DataFrame {i+1} saved to '{file_path}'", flush=True)
        else:
            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

    def all_node_metrics(self):

        cpu_endpoint = f"{self.base_url}/collect_metrics/all_node_metrics"
        response = requests.get(cpu_endpoint, params={'start': '-5m'})  

        if response.status_code == 200:

            print("DEU CERTO", flush=True)
            cpu_data = response.json()
            df_cpu = pd.DataFrame(cpu_data)
            print("CPU Metrics:", flush=True)
            print(df_cpu, flush=True)

        else:

            print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)


        print("Also saving on folder: main_tests_api")
        directory = '/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/'
        if os.path.exists(directory):
            print(f"Directory '{directory}' already exists.")
        else:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Directory '{directory}' created successfully.")
        df_cpu.to_csv(directory+"/"+"main_test_node.csv", index=False)

    def specific_logs(self):

        cpu_endpoint = f"{self.base_url}/collect_logs/specific_logs"
        response = requests.get(cpu_endpoint, params={'start': '-2m'})  

        if response.status_code == 200:

            print("DEU CERTO", flush=True)
            cpu_data = response.json()
            print("LOGS", cpu_data, flush= True)


        else:

            print(f"Failed to fetch LOGS: {response.status_code}", flush=True)


def main():

    #mudar para ip onde estiver a correr a API~, mudar diretorias nas funcoes para as certas
    base_url = "http://10.2.0.76:5010"
    call_api = CallAPI(base_url)
    conn = call_api.test_connection()
    if conn:
        print("test_connection:", conn, flush=True)
    call_api.all_container_metrics()
    print("middle", flush=True)
    #all_logs(base_url=base_url)
    print("CHEGUEI AO FIM", flush=True)
    '''df = pd.read_csv('/home/didioffside/MonitoringOneHost-main/collect_metrics/main_tests_api/main_test1.csv')
    column_names = df.columns.tolist()
    print(column_names, len(column_names))'''
    #all_container_metrics_start_stop(base_url, 1719235500, 1719236100, 0)

    #all_container_metrics_start_stop(base_url, 1719235500, 1719237299, 1)
    #all_container_metrics_start_stop(base_url, 1719237300, 1719239099, 2)
    #all_container_metrics_start_stop(base_url, 1719239100, 1719240899, 3)
    #all_container_metrics_start_stop(base_url, 1719240900, 1719242699, 4)
    #all_container_metrics_start_stop(base_url, 1719242700, 1719244499, 5)
    #all_container_metrics_start_stop(base_url, 1719244500, 1719246299, 6)
    #all_container_metrics_start_stop(base_url, 1719246300, 1719247200, 7)

    #all_container_cpu_metrics(base_url)
    #all_container_memory_metrics(base_url)
    #all_container_fs_metrics(base_url)
    #all_container_network_metrics(base_url)
    #all_container_spec_metrics(base_url)
    #all_node_metrics(base_url)
# Define the endpoint for CPU metrics



    '''influxdb_url = "http://172.17.0.1:5007"
    token = "InfluxDBToken"
    org = "MyOrg"
    bucket = "cadvisor_bucket"
    bucket_logs = "Second_Logs"
    bucket_networks = "Network"
    bucket_node = "node-exporter_bucket"
    time = 5'''

    #Select all container metrics
    #select_all_container_metrics(time, bucket, influxdb_url, token, org)

    #Select all host metrics
    #select_all_host_metrics(time, bucket_node, influxdb_url, token, org)

    #Function to return all packets
    #get_all_packets(time, bucket_networks, influxdb_url, token, org)

    #Function to get packet delays from ips
    #get_packet_delays(time, bucket_networks, influxdb_url, token, org, ip=True)

    #Function to get packet delays from protocols
    #get_packet_delays(time, bucket_networks, influxdb_url, token, org, protocol=True)

    #Function to get packet delays from ip and protocols
    #get_packet_delays(time, bucket_networks, influxdb_url, token, org, ip=True, protocol=True)



if __name__ == "__main__":

    main()