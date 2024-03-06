import pandas as pd
import requests
import numpy as np
import time


# Function to process a single translation ID
def process_translation_id(translation_id, page_link_tr, page_link_en, page_title_tr, page_title_en, is_scientific):
    row_of_nones = {'translation_id': translation_id, 'page_link_tr': page_link_tr, 'page_link_en': page_link_en,
                    'page_title_tr': page_title_tr, 'page_title_en': page_title_en, 'is_scientific': is_scientific,
                    'id_1': np.nan, 'id_2': np.nan, 'sequenceid': np.nan, 'source': np.nan, 'mt-engine': np.nan,
                    'mt-content': np.nan, 'target': np.nan}

    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&translationid={translation_id}&list=contenttranslationcorpora&striphtml=true"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code} for {translation_id}")
        return pd.DataFrame([row_of_nones])

    if 'query' not in response.json() or response.json()['query'] is None:
        print(f"Error: {response.status_code} for {translation_id}")
        return pd.DataFrame([row_of_nones])

    if 'contenttranslationcorpora' not in response.json()['query'] or response.json()['query'][
        'contenttranslationcorpora'] is None:
        print(f"Error: {response.status_code} for {translation_id}")
        return pd.DataFrame([row_of_nones])

    if 'sections' not in response.json()['query']['contenttranslationcorpora'] or \
            response.json()['query']['contenttranslationcorpora']['sections'] is None:
        print(f"Error: {response.status_code} for {translation_id}")
        return pd.DataFrame([row_of_nones])

    sections = response.json()['query']['contenttranslationcorpora']['sections']
    # AttributeError: 'list' object has no attribute 'keys'
    if not isinstance(sections, dict):
        print(f"Error: {response.status_code} for {translation_id}")
        return pd.DataFrame([row_of_nones])

    rows = []
    for key in sections.keys():
        row_i = {
            'translation_id': translation_id,
            'page_link_tr': page_link_tr,
            'page_link_en': page_link_en,
            'page_title_tr': page_title_tr,
            'page_title_en': page_title_en,
            'is_scientific': is_scientific,
            'id_1': f"{translation_id}/{key}",
            'id_2': key,
            'sequenceid': sections[key].get('sequenceid', np.nan),
            'source': sections[key]['source']['content'] if 'source' in sections[key] and sections[key][
                'source'] is not None else np.nan,
            'mt-engine': sections[key]['mt']['engine'] if 'mt' in sections[key] and sections[key][
                'mt'] is not None else np.nan,
            'mt-content': sections[key]['mt']['content'] if 'mt' in sections[key] and sections[key][
                'mt'] is not None else np.nan,
            'target': sections[key]['user']['content'] if 'user' in sections[key] and sections[key][
                'user'] is not None else np.nan
        }
        rows.append(row_i)

    return pd.DataFrame(rows)


def get_all_pairs_content():
    # Load the Excel file
    translation_ids_df = pd.read_excel('translation_pairs.xlsx')

    # Process each translation ID
    all_dataframes = []
    count = 0
    start_time = time.time()  # Record start time of the loop
    total_ids = len(translation_ids_df['translation_id'])

    for index, translation_id in enumerate(translation_ids_df['translation_id']):
        page_link_tr = \
            translation_ids_df[translation_ids_df['translation_id'] == translation_id]['page_link_tr'].values[0]
        page_link_en = \
            translation_ids_df[translation_ids_df['translation_id'] == translation_id]['page_link_en'].values[0]
        page_title_tr = \
            translation_ids_df[translation_ids_df['translation_id'] == translation_id]['page_title_tr'].values[0]
        page_title_en = \
            translation_ids_df[translation_ids_df['translation_id'] == translation_id]['page_title_en'].values[0]
        is_scientific = \
            translation_ids_df[translation_ids_df['translation_id'] == translation_id]['is_scientific'].values[0]

        df = process_translation_id(translation_id, page_link_tr, page_link_en, page_title_tr, page_title_en,
                                    is_scientific)
        all_dataframes.append(df)

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        items_processed = index + 1
        total_time_estimate = elapsed_time / items_processed * total_ids
        remaining_time = total_time_estimate - elapsed_time

        count += 1
        if count % 50 == 0 or count == total_ids:
            print(
                f"Processed {items_processed} out of {total_ids} translation IDs. Elapsed time: {elapsed_time:.2f} seconds. Remaining time: {remaining_time:.2f} seconds.")

    # Concatenate all DataFrames into one
    all_pairs = pd.concat(all_dataframes, ignore_index=True)

    # Save the data
    all_pairs.to_csv('all_pairs_with_content.csv', index=False)
    all_pairs.to_excel('all_pairs_with_content.xlsx', index=False)


def get_all_pairs_with_mt_and_target():
    all_pairs_with_content = pd.read_csv('all_pairs_with_content.csv')
    mt_and_target_present = all_pairs_with_content.dropna(subset=['target', 'mt-engine', 'mt-content'])
    mt_and_target_present.to_csv('mt_and_target_present.csv', index=False)
    mt_and_target_present.to_excel('mt_and_target_present.xlsx', index=False)


if __name__ == '__main__':
    # get_all_pairs_content()
    # get_all_pairs_with_mt_and_target()
    print("Done!")
