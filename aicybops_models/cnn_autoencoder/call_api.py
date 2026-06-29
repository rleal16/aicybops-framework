import requests
import pandas as pd
import json
import os

class CallAPI:

    def __init__(self, base_url):
        self.base_url = base_url

    def _login(self) -> str | None:
        login_endpoint = f"{self.base_url}/login"
        login_data = {
            'username': 'test_user_api',
            'password': 'test_password_api'
        }
        response = requests.post(login_endpoint, json=login_data)
        if response.status_code == 200:
            print("Successfully logged in", flush=True)
            return response.json().get('access_token')
        
        return None

        


    def _save_to_disk(self, df, directory, file_name):
        if os.path.exists(directory):
            print(f"Directory '{directory}' already exists.")
        else:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"Directory '{directory}' created successfully.")
        df.to_csv(directory+"/"+file_name, index=False)


        
    def all_container_metrics(self, start: str = '-120', save_to_disk: bool = False):
        # start is in seconds

        cpu_endpoint = f"{self.base_url}/collect_metrics/all_container_metrics"

        token = self._login()
        
        df_cpu = None

        if token:
            print(f"Token received: {token}", flush=True)
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(
                cpu_endpoint, 
                headers=headers, 
                params={'start': f'{start}s'},
                timeout=60
                )
            
            if response.status_code == 200:
                print("DEU CERTO", flush=True)
                cpu_data = response.json()
                df_cpu = pd.DataFrame(cpu_data)
                print("CPU Metrics:", flush=True)
                print(df_cpu, flush=True)

                if save_to_disk:
                    directory = './main_tests_api/'
                    file_name = "main_test.csv"
                    self._save_to_disk(df_cpu, directory, file_name)


            else:
                print(f"Failed to fetch CPU metrics: {response.status_code}", flush=True)

        
        return df_cpu

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

    def all_logs(self):
        login_endpoint = f"{self.base_url}/login"
        cpu_endpoint = f"{self.base_url}/collect_logs/all_logs"
        login_data = {
            'username': 'test_user_api',
            'password': 'test_password_api'
        }

        response = requests.post(login_endpoint, json=login_data)

        if response.status_code == 200:
            print("FIZ LOGIN", flush=True)
            token = response.json().get('access_token')
            print(f"Token received: {token}", flush=True)
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(cpu_endpoint, headers=headers, params={'start': '-2m'})  

        if response.status_code == 200:

            print("DEU CERTO", flush=True)
            cpu_data = response.json()
            print("LOGS", cpu_data, flush= True)

        else:

            print(f"Failed to fetch LOGS: {response.status_code}", flush=True)

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