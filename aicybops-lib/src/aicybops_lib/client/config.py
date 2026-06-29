from dataclasses import dataclass

@dataclass
class ClientConfig:
    base_url: str
    max_retries: int = 5
    retry_delay: int = 2
    timeout: int = 30
    verify_ssl: bool = True
