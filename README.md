# Application Log Generation

Creates realistic application logs for testing data pipelines, analytics dashboards, and ML models. Uses configurable user personas and journey definitions to simulate traffic patterns.

## Setup

Start the database:

```bash
docker compose up -d
```

## Generate Logs

Jupyter notebook:

```python
!python generator.py generate --consumer consumer.yml --personas personas.yml
```

Output goes to `./out/` with CSVs for iteration visibility:
- `dim_users.csv` - user dimension table
- `application_logs.csv` - request logs
- `manifest.json` - generation metadata

## Sample output
<img width="1229" height="721" alt="image" src="https://github.com/user-attachments/assets/81234cc7-c43a-4b18-b084-b8bf75f52ca5" />
<img width="1090" height="246" alt="image" src="https://github.com/user-attachments/assets/7eab0c46-ba20-4caa-8f70-6e325670eaa4" />

