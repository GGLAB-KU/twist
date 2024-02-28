import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json


def create_session():
    return requests.Session()


def fetch_random_terms(session, base_url, num_terms=10000):
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

    with open('random_terms.json', 'w', encoding='utf-8') as f:
        json.dump(random_terms, f, ensure_ascii=False, indent=4)

    print(f"Collected {len(random_terms)} random terms.")
    return random_terms


def normalize_term(term):
    replacements = {'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u', 'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O',
                    'Ş': 'S', 'Ü': 'U'}
    normalized = ''.join(replacements.get(c, c) for c in term).lower().replace(' ', '-')
    return normalized


def process_terms(terms):
    data = [{'term_in_turkish': term, 'url_part': normalize_term(term)} for term in terms]
    df_terms = pd.DataFrame(data)
    print(df_terms)
    return df_terms


def fetch_term_details(session, df_terms, base_url):
    all_terms = []
    for i, row in enumerate(df_terms.itertuples(index=False), 1):
        if i % 100 == 0:
            print(f"Processed {i} terms.")
        try:
            response = session.get(f"{base_url}{row.url_part}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            main_search_tag = soup.find('single-term-container')
            if main_search_tag and ':term' in main_search_tag.attrs:
                term = json.loads(main_search_tag[':term'].strip("'"))
                all_terms.append(term)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        time.sleep(0.1)

    new_df = pd.DataFrame(all_terms)
    new_df.to_csv('new_terms.csv', index=False, encoding='utf-8')
    print(new_df)
    return new_df


def remove_duplicates_and_save(df):
    df = df.drop_duplicates(subset='id', keep='first')
    df = df.sort_values(by='id').reset_index(drop=True)
    df.to_csv('unique.csv', index=False)
    df.to_excel('unique.xlsx', index=False)
    return df


def main():
    base_url = "https://terimler.org"
    num_terms = 100
    session = create_session()
    random_terms = fetch_random_terms(session, base_url, num_terms)
    terms = process_terms(random_terms)
    df_terms = fetch_term_details(session, terms, base_url + "/terim/")
    df_cleaned = remove_duplicates_and_save(df_terms)
    print("Data processing complete.")


if __name__ == "__main__":
    main()
