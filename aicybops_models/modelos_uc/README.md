# Modelos Exemplos



## Ficheiros

### 1. Dados

Há 6 ficheiros dos datasets utilizados para treino e teste dos modelos:
- **`all_runs.csv`**: Contém as samples de todos experimentos (attack & gold runs). Não uso em nenhum modelo, mas fica aqui disponível.

- **`golden_runs.csv`**: Contém as samples de todas as golden runs. (Treino modelo unsupervised)
- **`attack_runs.csv`**: Contém as samples de todas as attack runs. (Teste modelo unsupervised)

- **`training_data.csv`**: Contém 70% de todas as samples. (Treino modelo supervised)
- **`testing_data.csv`**: Contém 30% de todas as samples. (Teste modelo supervised)


### 2. Scripts simples dos modelos

- **`anomalydetection_ae.py`**: Um Autoencoder que treina com dados normais e testa com os dados das attack runs que podem ter samples de ataque.
- **`supervised_xgboost.py`**: Modelo XGBoost que classifica as samples em normal (0) ou ataque (1).

