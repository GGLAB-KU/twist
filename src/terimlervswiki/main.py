import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import wikipediaapi


def simulate_coverage(total_terms, total_draws, num_simulations):
    np.random.seed(0)  # For reproducibility
    unique_counts = []

    for _ in range(num_simulations):  # Number of simulations
        unique_numbers = set()
        draws = np.random.choice(range(1, total_terms + 1), total_draws, replace=True)
        unique_numbers.update(draws)
        unique_counts.append(len(unique_numbers))

    # Calculating the average number of unique integers across all simulations
    expected_mean = np.mean(unique_counts)
    expected_coverage = expected_mean / total_terms
    print(f'Expected mean: {expected_mean:.2f}')
    print(f'Expected coverage: {expected_coverage:.2%}')


def create_session():
    """Create and return a new web session."""
    return requests.Session()


def fetch_random_terms(session, base_url, num_terms=10):
    """Fetch a list of random terms from the given base URL."""
    random_terms = []
    for i in range(num_terms):
        if i % 100 == 0:
            print(f"Collected {i} random terms.")
        try:
            response = session.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            main_search_tag = soup.find('main-search')
            if main_search_tag and ':random-term' in main_search_tag.attrs:
                random_term = main_search_tag[':random-term'].strip("'")
                random_terms.append(random_term)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(0.1)
    # Remove duplicates
    random_terms = list(set(random_terms))
    return random_terms


def normalize_term(term):
    """Normalize special characters in a term and return its URL part."""
    replacements = {"â": "a", 'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u', 'Ç': 'C', 'Ğ': 'G', 'İ': 'I',
                    'Ö': 'O', 'Ş': 'S', 'Ü': 'U', "'": "", "’": "", }
    # replace all punctiation with empty string
    term = term.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return ''.join(replacements.get(c, c) for c in term).lower().replace(' ', '-')


def add_url(terms):
    """Add URL parts to the terms and return a DataFrame."""
    data = [{'term_in_turkish': term, 'url_part': normalize_term(term)} for term in terms]
    return pd.DataFrame(data)


def get_term_details(session, url):
    """Fetch and parse term details from the given URL."""
    try:
        response = session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def parse_term_details(soup):
    """Extract term details from BeautifulSoup object."""
    main_search_tag = soup.find('single-term-container')
    if main_search_tag and ':term' in main_search_tag.attrs:
        return json.loads(main_search_tag[':term'].strip("'"))
    return None


def process_term(term):
    """Yield processed data rows for a single term."""
    for definition, translation in zip(term['definitions'], term['translations']):
        synonyms_list = translation['synonym'].split(";")
        number_of_synonyms, synonyms = process_synonyms(synonyms_list)
        en, number_of_en = process_language_translation(translation['en'])
        de, number_of_de = process_language_translation(translation['de'])
        fr, number_of_fr = process_language_translation(translation['fr'])
        lat, number_of_lat = process_language_translation(translation['lat'])
        yield {
            'id': term['id'], 'name': term['name'], 'slug': term['slug'],
            'number_of_definitions': len(term['definitions']),
            'scope': definition['scope'], 'content': definition['content'], 'number_of_synonyms': number_of_synonyms,
            'synonyms': synonyms, 'number_of_en': number_of_en, 'number_of_de': number_of_de,
            'number_of_fr': number_of_fr, 'number_of_lat': number_of_lat, 'en': en, 'de': de, 'fr': fr, 'lat': lat
        }


def process_synonyms(synonyms):
    """Process synonyms to count and format them properly."""
    synonyms = list(filter(None, synonyms))  # Remove empty strings
    return len(synonyms), ";".join(synonyms)


def process_language_translation(translation):
    """Process language translations to count and format them properly."""
    translations = list(filter(None, translation.split(";")))
    return ";".join(translations), len(translations)


def fetch_and_process_terms(session, terms, base_url):
    """Fetch and process term details, returning a DataFrame of the results."""
    df_rows = []
    for i, term in enumerate(terms, 1):
        if i % 100 == 0:
            print(f"Processed {i} terms.")
        soup = get_term_details(session, f"{base_url}{term['url_part']}")
        if soup:
            term_details = parse_term_details(soup)
            if term_details:
                df_rows.extend(process_term(term_details))
        time.sleep(0.1)

    return pd.DataFrame(df_rows)


def get_random_terms(base_url, num_terms, session, input_path):
    random_terms = fetch_random_terms(session, base_url, num_terms)
    with open(input_path, 'w') as f:
        json.dump(random_terms, f, indent=4, ensure_ascii=False)


def process_random_terms(input_path, base_url, session, output_path):
    with open(input_path, 'r') as f:
        random_terms = json.load(f)
    terms = add_url(random_terms)
    df_terms = fetch_and_process_terms(session, terms.to_dict('records'), base_url + "/terim/")
    df_terms = df_terms.sort_values(by='id')
    df_terms = df_terms.reset_index(drop=True)
    print(f"Number of terms: {len(df_terms)}")
    print(f"Number of unique terms: {len(df_terms['id'].unique())}")

    df_filtered = df_terms[(df_terms['number_of_definitions'] == 1) & (df_terms['number_of_synonyms'] == 0) & (
            df_terms['number_of_en'] == 1)]

    print(
        f"Number of filtered terms: {len(df_filtered)}, which is {len(df_filtered) / len(df_terms['id'].unique()) * 100:.2f}% of "
        f"the total terms.")
    df_filtered.to_csv(output_path + "_one2one.csv", index=False)
    df_terms.to_csv(output_path + "_all.csv", index=False)

    df_filtered.to_excel(output_path + "_one2one.xlsx", index=False)
    df_terms.to_excel(output_path + "_all.xlsx", index=False)


def terimler_org_part(base_url, num_terms, session, input_path, output_path):
    get_random_terms(base_url, num_terms, session, input_path)
    process_random_terms(input_path, base_url, session, output_path)


def fetch_wiki_data(term, id, lang='en'):
    wiki_wiki = wikipediaapi.Wikipedia('some user agent - 1', lang)
    try:
        page = wiki_wiki.page(term)
        if page.exists():
            # get the Turkish page if it exists
            tr_page = page.langlinks.get('tr')
            if tr_page is not None:
                return {'id': id,  # Add the unique identifier to the result
                        lang + '_exists': True,
                        lang + '_pageid': page.pageid,
                        lang + '_title': page.title,
                        lang + '_fullurl': page.fullurl,
                        lang + '_length': page.length,
                        lang + '_summary': page.summary,
                        'tr_exists': True,
                        'tr_pageid': tr_page.pageid,
                        'tr_title': tr_page.title,
                        'tr_fullurl': tr_page.fullurl,
                        'tr_length': tr_page.length,
                        'tr_summary': tr_page.summary,
                        'category': 3.0}
            else:
                return {
                    'id': id,  # Add the unique identifier to the result
                    lang + '_exists': True,
                    lang + '_pageid': page.pageid,
                    lang + '_title': page.title,
                    lang + '_fullurl': page.fullurl,
                    lang + '_length': page.length,
                    lang + '_summary': page.summary,
                    'tr_exists': False,
                    'category': 2.0}
        else:
            return {'id': id,
                    lang + '_exists': False,
                    'tr_exists': False,
                    'category': 1.0}

    except Exception as e:
        print(f"Error fetching data for term: {term}, Error: {e}")
        return None


def wikipedia_part(input_path, output_path):
    terms_df = pd.read_csv(input_path)
    processed_terms = 0

    results = []
    for index, row in terms_df.iterrows():
        term = row['en']
        id = row['id']
        result = fetch_wiki_data(term, id)
        if result is not None:
            results.append(result)

        processed_terms += 1
        if processed_terms % 100 == 0:
            print(f"Processed {processed_terms} terms.")

    df = pd.DataFrame(results)
    df.to_csv(output_path + ".csv", index=False)
    df.to_excel(output_path + ".xlsx", index=False)


def merge_terimler_org_and_wikipedia(terimler_org_path, wikipedia_path):
    terimler_org_df = pd.read_csv(terimler_org_path)
    wikipedia_df = pd.read_csv(wikipedia_path)
    merged_df = terimler_org_df.merge(wikipedia_df, on='id', how='left')
    merged_df = merged_df.drop_duplicates(subset='id', keep=False)
    merged_df.to_csv("terms/merged/merged.csv", index=False)
    merged_df.to_excel("terms/merged/merged.xlsx", index=False)


if __name__ == "__main__":
    base_url = "https://www.terimler.org"
    num_samples = 150000
    input_path_terimler_org = "terms/terimler_org/random_terms.json"
    output_path_terimler_org = "terms/terimler_org/terimler_org"
    input_path_wikipedia = output_path_terimler_org + "_one2one.csv"
    output_path_wikipedia = "terms/wikipedia/wikipedia"

    simulate_coverage(51000, num_samples, 10)

    session = create_session()
    terimler_org_part(base_url, num_samples, session, input_path_terimler_org, output_path_terimler_org)

    wikipedia_part(input_path_wikipedia, output_path_wikipedia)

    merge_terimler_org_and_wikipedia(input_path_wikipedia, output_path_wikipedia + ".csv")
