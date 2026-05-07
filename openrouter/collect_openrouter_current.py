import requests
import json
import pandas as pd
import time

from set_globals import *

snapshot_folder = openrouter_api_dir
model_endpoints_folder = openrouter_api_dir / 'model_endpoints'

# other possible URLs:
# https://openrouter.ai/api/v1/parameters/:author/:slug # seems to require authentication
# https://openrouter.ai/api/v1/activity # seems to require authentication

# need to loop over all models for this one
# https://openrouter.ai/api/v1/models/:author/:slug/endpoints

# probably just want to loop through api/v1 and api/frontend for all endpoints

endpoints = ['all-providers',
             'provider-filters',
             'models',
             'models/count',
             'embeddings/models',
             'providers',
             'models/find']

timestamp = int(time.time())
model_list = []
for api_name in ['v1', 'frontend']:
    for e in endpoints:
        url = f'https://openrouter.ai/api/{api_name}/{e}'

        filename = f'openrouter_api_{api_name}_{e.replace("/", "_")}_{timestamp}.json'
        snapshot_path = Path(snapshot_folder / filename)

        if snapshot_path.exists():
            print(f'Already downloaded {snapshot_path}, skipping')
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()

            snapshot_path.write_bytes(response.content)

            if api_name == 'v1' and e == 'models':
                model_list = [m['canonical_slug'] for m in json.loads(response.content)['data']]

            print(f'Downloaded {url}')
        except Exception as ex:
            print(f'Error downloading {url}: {ex}')

        # One request every 5 seconds
        time.sleep(5)

for model in model_list:
    url = f'https://openrouter.ai/api/v1/models/{model}/endpoints'
    model_path = Path(model_endpoints_folder / f'endpoints_{model.replace("/", "_")}_{timestamp}.json')

    if model_path.exists():
        print(f'Already got endpoint data for {model}, skipping')
        continue

    try:
        response = requests.get(url)
        response.raise_for_status()
        model_path.write_bytes(response.content)
        print(f'Downloaded {url}')
    except Exception as ex:
        print(f'Error downloading {url}: {ex}')

    # One request every 5 seconds
    time.sleep(5)

