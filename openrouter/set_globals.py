import datetime
from pathlib import Path

# Root of the fm-econ-growth project
project_dir = Path(__file__).parent.parent

# Data directories
data_dir     = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
temp_dir     = data_dir / 'temp'

# OpenRouter API snapshot directory (used by collect_openrouter_current.py)
openrouter_api_dir = raw_data_dir / 'openrouter_snapshots' / 'api'

# Ensure directories exist
for _d in [data_dir, raw_data_dir, temp_dir, openrouter_api_dir,
           openrouter_api_dir / 'model_endpoints',
           raw_data_dir / 'openrouter_snapshots' / 'wayback']:
    _d.mkdir(parents=True, exist_ok=True)
