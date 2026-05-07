import numpy as np
import pandas as pd
import os
import re
import json
import json_repair

from set_globals import *

snapshot_folder = raw_data_dir / 'openrouter_snapshots'
wayback_folder = snapshot_folder / 'wayback'
api_folder = snapshot_folder / 'api'

# This function searches for matching curly braces to extract JSON objects from possibly poorly formed strings
def get_objs(s):
    obj_list = []
    left_indices = [i for i, c in enumerate(s) if c == '{']
    for l in left_indices:
        depth = 0
        for j in range(l, len(s)):
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        # Found the right curly brace corresponding to l,
                        # so try to parse the object.
                        # Then, break and move on to the next left curly brace
                        obj = json_repair.loads(s[l:j+1])
                        obj_list.append(obj)
                        break
                    except json.JSONDecodeError as e:
                        # If we can't load the object, the inner for loop continues.
                        # It's possible that depth with reach zero again but the JSON will likely be malformed at that point.
                        print(f'JSON decode error {e}')
                        pass

    # Extract objects from lists
    extracted_from_lists = []
    for o in obj_list:
        if isinstance(o, list):
            extracted_from_lists.extend([item for item in o if isinstance(item, dict)])

    obj_list.extend(extracted_from_lists)

    return obj_list

model_list = []
usage_list = []

snapshot_paths = []
snapshot_paths += list(wayback_folder.glob('openrouter_models_*'))

snapshot_paths += list(api_folder.glob('openrouter_api_frontend_models_[0-9]*.json')) # seems to have same format as wayback JSON
snapshot_paths += list(api_folder.glob('openrouter_api_frontend_models_find_[0-9]*.json'))
snapshot_paths += list(api_folder.glob('openrouter_api_v1_models_[0-9]*.json'))
snapshot_paths += list((api_folder / 'model_endpoints').glob('endpoints_*.json'))

for snapshot_path in snapshot_paths:
    print(f'Processing {snapshot_path}')

    # need to change this for API JSON files
    if snapshot_path.parent.name == 'wayback':
        snapshot_date = datetime.datetime.strptime(snapshot_path.stem.split('_')[-1], '%Y%m%d%H%M%S')
    else:
        snapshot_date = datetime.datetime.fromtimestamp(int(snapshot_path.stem.split('_')[-1]))

    try:
        if snapshot_path.suffix == '.json':
            if snapshot_path.parent.name == 'model_endpoints':
                raw = json.loads(snapshot_path.read_text())['data']
                stem = snapshot_path.stem
                middle = re.sub(r'^endpoints_', '', stem)
                middle = re.sub(r'_\d+$', '', middle)
                author, model_name = middle.split('_', 1)
                openrouter_id = f'{author}/{model_name}'
                endpoints = raw.get('endpoints', [])
                if endpoints:
                    ep = endpoints[0]
                    model_data = {
                        'datetime': snapshot_date,
                        'openrouter_id': openrouter_id,
                        'openrouter_author': author,
                        'openrouter_model_name': model_name,
                        'context_length': ep.get('context_length'),
                    }
                    for k, v in ep.get('pricing', {}).items():
                        model_data[f'{k}_price'] = v
                    model_list.append(model_data)
                continue
            snapshot_data = json.loads(snapshot_path.read_text())['data']
        elif snapshot_path.suffix == '.html':
            # The raw HTML text contains JSON with a list of models
            pattern = r'self\.__next_f\.push\((.+?)\)</script>'
            for match_text in re.findall(pattern, snapshot_path.read_text(encoding = 'utf-8'), re.DOTALL):
                if 'context_length' in match_text:
                    # Found the instance of __next_f.push that contains the model list JSON as its argument
                    break

            # TODO getting list index out of range for a few snapshots, check this
            model_dict = json.loads(json.loads(match_text)[1][2:])[3]['children'][0][3]['children'][3]['children'][3]['children'][3]
            if len(model_dict.keys()) > 3:
                print('Model dictionary has more than 3 keys, here they are:')
                print(model_dict.keys())
                print()

            snapshot_data = model_dict['featuredTextModels']
            snapshot_data.extend(model_dict['featuredMediaModels'])

    except Exception as e:
        print(f'Failed to load models from snapshot with error {e}')
        continue

    if 'find' in snapshot_path.stem:
        # Store usage data: snapshot_data['analytics'] is a dict of dicts
        usage_list.extend([{'datetime': snapshot_date,
                            'openrouter_id': k,
                            **v} for k, v in snapshot_data['analytics'].items()])
    else:
        for m in snapshot_data:
            # Loop over models in the snapshot JSON

            openrouter_id = None
            if 'canonical_slug' in m.keys():
                # Use canonical_slug (dash+date form) as base ID for consistency with usage analytics,
                # appending any variant suffix (e.g. ':thinking') from the id field
                variant = m['id'].split(':', 1)[1] if ':' in m['id'] else ''
                openrouter_id = m['canonical_slug'] + (':' + variant if variant else '')
            elif 'permaslug' in m.keys():
                # Wayback format: same logic using permaslug + variant from slug
                variant = m['slug'].split(':', 1)[1] if ':' in m['slug'] else ''
                openrouter_id = m['permaslug'] + (':' + variant if variant else '')
            elif 'slug' in m.keys():
                openrouter_id = m['slug']

            if not openrouter_id:
                continue

            op_author = openrouter_id.split('/')[-2]
            op_model_name = openrouter_id.split('/')[-1]

            model_data = {'datetime': snapshot_date,
                          'openrouter_id': openrouter_id,
                          'openrouter_author': op_author,
                          'openrouter_model_name': op_model_name,
                          'context_length': m['context_length']}

            huggingface_id = None
            if 'hf_slug' in m.keys():
                huggingface_id = m['hf_slug']
            elif 'hugging_face_id' in m.keys():
                huggingface_id = m['hugging_face_id']

            if huggingface_id:
                model_data['huggingface_id'] = huggingface_id

            endpoint = None
            if 'endpoint' in m.keys():
                endpoint = m['endpoint']

            pricing = {}
            if 'pricing' in m.keys():
                pricing = m['pricing']
            elif endpoint:
                pricing = endpoint['pricing']

            for k in pricing.keys():
                model_data[f'{k}_price'] = pricing[k]

            if 'hidden' in m.keys():
                model_data['hidden'] = m['hidden']

            if endpoint:
                for k in ['is_hidden', 'is_deranked', 'is_disabled']:
                    if k in endpoint.keys():
                        model_data[f'endpoint_{k}'] = endpoint[k]

                model_data['endpoint_provider'] = endpoint['provider_name']

            model_list.append(model_data)

df_model = pd.DataFrame(model_list)

# Two cleaning levels:
#  - _strip_date: strips trailing all-digit date suffixes only; preserves :variant. Used to
#    derive the public-facing `openrouter_id` (clean form, with variant intact).
#  - _hf_match_key: aggressive strip (date + variant). Internal use only — broadcasts a single
#    huggingface_id across rows whose IDs differ only by date/variant.
# Longest date patterns listed first so e.g. "...-2024-11-20" strips fully rather than matching
# the shorter "-11-20" tail.
date_suffix_re = re.compile(r'-(\d{4}-\d{2}-\d{2}|\d{2}-\d{4}|\d{2}-\d{2}|\d{8}|\d{6}|\d{4})$')

def _strip_date(x):
    base, sep, variant = x.partition(':')
    return date_suffix_re.sub('', base) + (sep + variant if sep else '')

def _hf_match_key(x):
    return date_suffix_re.sub('', x.split(':', 1)[0])

# Keep raw IDs as the row-level key so raw IDs that share a clean form but list different
# prices stay as separate rows (revenue downstream multiplies the actual price by the actual
# tokens for that price point).
df_model = df_model.rename(columns = {'openrouter_id': 'openrouter_id_uncleaned'})

price_cols = [c for c in df_model.columns if c.endswith('_price')]
for c in price_cols:
    df_model[c] = pd.to_numeric(df_model[c], errors = 'coerce')
    df_model.loc[df_model[c] < 0, c] = np.nan
df_usage = pd.DataFrame(usage_list)
df_usage = df_usage.rename(columns = {'openrouter_id': 'openrouter_id_uncleaned'})
df_usage = df_usage.drop(columns = 'date') # Unclear what exactly this is, just use the actual snapshot date for now
df_usage['total_tokens'] = df_usage['total_prompt_tokens'] + df_usage['total_completion_tokens']

df_model['date'] = pd.to_datetime(df_model['datetime'].dt.date, utc = True)
df_usage['date'] = pd.to_datetime(df_usage['datetime'].dt.date, utc = True)
df_model = df_model.drop(columns = 'datetime')
df_usage = df_usage.drop(columns = 'datetime')

df = pd.merge(left = df_model,
              right = df_usage,
              on = ['openrouter_id_uncleaned', 'date'],
              how = 'outer')

agg_dict = {}
for c in df.columns:
    if c in ('openrouter_id_uncleaned', 'date'):
        continue

    agg_dict[c] = 'first'
    try:
        df[c] = pd.to_numeric(df[c])
        # Prices: take the cheapest endpoint observed when multiple snapshots collide on the
        # same (raw_id, date). Token counts and other numeric columns: take max (cumulative
        # reports — the highest observation reflects the most-complete snapshot).
        agg_dict[c] = 'min' if c.endswith('_price') else 'max'
    except (ValueError, TypeError) as e:
        continue

# There may be multiple snapshots from a single date.
# I want one observation per model per date.
df = df.groupby(['openrouter_id_uncleaned', 'date']).agg(agg_dict).reset_index()

# Create panel structure
model_dates = df.groupby('openrouter_id_uncleaned')['date'].apply(lambda g: pd.date_range(g.min(), g.max(), freq = 'D')).reset_index().explode('date')

df = pd.merge(left = model_dates,
              right = df,
              how = 'left',
              on = ['openrouter_id_uncleaned', 'date'])

df = df.sort_values(by = ['openrouter_id_uncleaned', 'date']).reset_index(drop = True)

# Cleaned ID: date suffix stripped, :variant preserved. Annotation column for downstream
# market-share consolidation; multiple uncleaned rows can share one cleaned value.
df['openrouter_id'] = df['openrouter_id_uncleaned'].apply(_strip_date)

# Broadcast huggingface_id across all rows whose IDs differ only by date/variant. The key
# is computed inline (not persisted) — variants share an underlying HF model, so :thinking
# and :beta inherit the same hf_slug as the bare model.
hf_key = df['openrouter_id_uncleaned'].apply(_hf_match_key)
df['huggingface_id'] = df.groupby(hf_key)['huggingface_id'].transform('first')

df['huggingface_author'] = df['huggingface_id'].str.split('/').str[-2]
df['huggingface_model_name'] = df['huggingface_id'].str.split('/').str[-1]

df.to_parquet(data_dir / 'openrouter_panel.parquet', index = False)
df.to_csv(temp_dir / 'openrouter_panel.csv', index = False)
