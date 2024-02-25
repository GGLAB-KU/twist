import os
import os.path

import requests
from bs4 import BeautifulSoup

from src.utils import unzip_gz, download


class ContentDumpDownloader:
    def __init__(self, num_latest_dumps_to_use: int = 2):
        self.url = 'https://dumps.wikimedia.org/other/contenttranslation/'
        self.num_latest_dumps_to_use = num_latest_dumps_to_use

    def download_dumps(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            hrefs = [a['href'] for a in soup.find_all('a')]
            valid_refs = list(filter(lambda x: len(x) > 2, map(lambda x: x.split('/')[0], hrefs)))
            valid_refs = sorted(valid_refs, reverse=True)
            for href in valid_refs[:self.num_latest_dumps_to_use]:
                print('processing: ', href)
                self._get_links_for_date_ref(href)
        else:
            print(f"Failed to retrieve content from {self.url}, status code {response.status_code}")

    def _get_links_for_date_ref(self, date_ref):
        date_path = os.path.join(self.url, date_ref)
        response = requests.get(date_path)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            hrefs = [a['href'] for a in soup.find_all('a')]
            valid_refs = list(filter(lambda x: 'en2tr' in x, map(lambda x: x.split('/')[0], hrefs)))
            valid_refs = [x for x in valid_refs if 'html' in x or 'text.json' in x]

            if len(valid_refs) != 2:
                print('text and html should exist! moving on...')
                return

            for href in valid_refs:
                file_url = os.path.join(date_path, href)
                out_location = f'../data/{date_ref}/{href}'
                json_location = out_location.replace('.gz', '')
                os.makedirs(f'../data/{date_ref}', exist_ok=True)

                if os.path.exists(out_location):
                    print(out_location, 'exists...')
                else:
                    download(file_url, out_location)

                print('unzipping downloaded file...')
                if os.path.exists(json_location):
                    print(json_location, 'exists...')
                else:
                    unzip_gz(out_location, json_location)

        else:
            print(f"Failed to retrieve content from {self.url}, status code {response.status_code}")
            return

        print('completed...')


if __name__ == '__main__':
    dump_downloader = ContentDumpDownloader(num_latest_dumps_to_use=2)
    dump_downloader.download_dumps()
