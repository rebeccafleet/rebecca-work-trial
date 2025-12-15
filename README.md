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

