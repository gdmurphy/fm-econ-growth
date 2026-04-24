import time
import requests
import re
import json
import pandas as pd
from waybackpy import WaybackMachineCDXServerAPI
import json_repair

from openrouter.set_globals import *

decoder = json.JSONDecoder()

snapshot_folder = raw_data_dir / 'openrouter_snapshots' / 'wayback'

# May also want to get
# https://openrouter.ai/api/frontend/all-providers
# https://openrouter.ai/api/frontend/provider-filters
# https://openrouter.ai/api/v1/models
# https://openrouter.ai/api/v1/[etc]
# The api/v1 endpoints seem to have fewer snapshots on the Wayback machine

urls = ['https://openrouter.ai/models',
        'https://openrouter.ai/api/frontend/models',
        'https://openrouter.ai/api/frontend/models/find']
user_agent = 'Mozilla/5.0'

for url in urls:
    print(f'Downloading snapshots of {url}')
    cdx = WaybackMachineCDXServerAPI(url, user_agent)
    total_requests = 0
    for snapshot in cdx.snapshots():
        if url == 'https://openrouter.ai/api/frontend/models':
            snapshot_path = Path(snapshot_folder / f'openrouter_models_{snapshot.timestamp}.json')
        elif url == 'https://openrouter.ai/api/frontend/models/find':
            snapshot_path = Path(snapshot_folder / f'openrouter_models_find_{snapshot.timestamp}.json')
        elif url == 'https://openrouter.ai/models':
            snapshot_path = Path(snapshot_folder / f'openrouter_models_{snapshot.timestamp}.html')

        if snapshot_path.exists():
            print(f'Already downloaded {snapshot_path}, skipping')
            continue

        # Avoid getting rate-limited by Wayback
        total_requests += 1
        if total_requests % 5 == 0:
            time.sleep(30)
        time.sleep(5)

        archive_url = snapshot.archive_url
        try:
            response = requests.get(archive_url)
            response.raise_for_status()

            snapshot_path.write_bytes(response.content)
            print(f'Downloaded {archive_url}')
        except Exception as e:
            print(f'Error downloading {archive_url}: {e}')
            continue

    print()

print('Done downloading snapshots.\n\n')

