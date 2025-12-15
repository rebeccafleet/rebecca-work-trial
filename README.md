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
<img width="1090" height="246" alt="image" src="https://github.com/user-attachments/assets/455e1d40-2040-4ec4-bc97-86865385f2e9" />
<img width="1217" height="698" alt="image" src="https://github.com/user-attachments/assets/c27355cd-b06a-41bd-a90f-121e647ca0cf" />

