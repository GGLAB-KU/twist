import json
import os

import ijson
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


class ContentDumpReader:
    def __init__(self, data_dir: str = '../data', num_dumps: int = 1):
        self.data_dir = data_dir
        self.content_dumps = []
        items = os.listdir(self.data_dir)
        for item in items[:num_dumps]:
            self.content_dumps.append(item)

        self.csv_dict = {}

        for dump_alias in self.content_dumps:
            csv_path = self._dump_to_csv(dump_alias)
            self.csv_dict[dump_alias] = csv_path

    def count_without_mts(self, dump_alias: str):
        #  number of from scratch targets (without MT)
        csv_path = self.csv_dict[dump_alias]
        content_df = pd.read_csv(csv_path, dtype={'id_2': pd.StringDtype()})

        total_num_documents = 0
        total_num_sections = len(content_df)
        num_docs_without_mts = 0
        num_sections_without_mts = 0

        for name, group in tqdm(content_df.groupby('id_1'), desc='processing documents...'):
            # TODO: @gsoykan - create a document object - fully concat...
            total_num_documents += 1
            for row in group.itertuples(index=False):
                if pd.isna(row.mt):
                    num_sections_without_mts += 1

            if num_sections_without_mts == len(group):
                num_docs_without_mts += 1

        return {
            'total_num_documents': total_num_documents,
            'total_num_sections': total_num_sections,
            'num_docs_without_mts': num_docs_without_mts,
            'num_sections_without_mts': num_sections_without_mts
        }

    def _dump_to_csv(self, dump_alias: str) -> str:
        dump_folder_path = os.path.join(self.data_dir, dump_alias)
        files = os.listdir(dump_folder_path)
        files = list(filter(lambda f: f.endswith('.json'), files))
        text_json_path = None
        html_json_path = None
        for file in files:
            file_path = os.path.join(dump_folder_path, file)
            if 'text' in file_path:
                text_json_path = file_path
            elif 'html' in file_path:
                html_json_path = file_path

        csv_path = os.path.join(dump_folder_path, f'items.csv')
        if os.path.exists(csv_path):
            print('csv exists ', csv_path)
            return csv_path

        # data/20240216/ text json 'ında hata var elle düzeltmek lazım
        # ijson.common.IncompleteJSONError: parse error: unallowed token at this point in JSON text
        #           : "[1]"         }     },     ,     {         "id": "431229/c
        #                      (right here) ------^
        # .html i de bozuk, şu komutu çalıştırmak lazım => sed -i '425312d' cx-corpora.en2tr.html.json

        # TODO: @gsoykan - diğerinde de var sanırım
        #  fix it...
        # ijson.common.IncompleteJSONError: parse error: unallowed token at this point in JSON text
        #           p></p>"         }     },     ,     {         "id": "431229/c
        #                      (right here) ------^

        html_dict = {}
        with open(html_json_path, 'r') as file:
            array_items = ijson.items(file, 'item')
            for item in tqdm(array_items, desc='processing html'):
                html_dict[item['id']] = item

        csv_items = []

        with open(text_json_path, 'r') as file:
            array_items = ijson.items(file, 'item')
            for item in tqdm(array_items, desc='processing json'):
                item_id = item['id']
                html_pair = html_dict.get(item_id, None)
                if html_pair is not None:
                    html_source, html_mt, html_target = (html_pair.get('source') or {}).get('content'), (
                            html_pair.get('mt') or {}).get(
                        'content'), (html_pair.get('target') or {}).get('content')
                else:
                    html_source, html_mt, html_target = None, None, None
                html_source = html_source if html_source != '' else None
                html_mt = html_mt if html_mt != '' else None
                html_target = html_target if html_target != '' else None
                html_source_hyperlinks = None
                html_target_hyperlinks = None

                def get_hyperlinks(html_text):
                    s = BeautifulSoup(html_text, 'lxml')
                    hyperlinks = s.find_all('a')
                    hyperlinks = list(filter(lambda x: x.img is None and
                                                       x.get('href') is not None and
                                                       '#cite' not in x.get('href', ''), hyperlinks))
                    hyperlinks = list(map(lambda x: {
                        'href': x['href'],
                        'id': x.get('id'),
                        'data_linkid': x.get('data-linkid'),
                        'rel': x.get('rel'),
                        'string': x.string,
                        'text': x.text,

                    }, hyperlinks))
                    return hyperlinks

                if html_source is not None:
                    html_source_hyperlinks = get_hyperlinks(html_source)
                    html_source_hyperlinks = json.dumps(html_source_hyperlinks)
                if html_target is not None:
                    html_target_hyperlinks = get_hyperlinks(html_target)
                    html_target_hyperlinks = json.dumps(html_target_hyperlinks)

                item_source, item_mt, item_target = (item.get('source') or {}).get('content'), (
                        item.get('mt') or {}).get(
                    'content'), (item.get('target') or {}).get('content')

                item_id_first, item_id_second = item_id.split('/')
                csv_item = [
                    item_id,
                    item_id_first,
                    item_id_second,
                    item_source if item_source != '' else None,
                    item_mt if item_mt != '' else None,
                    item_target if item_target != '' else None,
                    html_source_hyperlinks,
                    html_target_hyperlinks
                ]

                csv_items.append(csv_item)

        print('saving csv...')
        df = pd.DataFrame(csv_items, columns=['id',
                                              'id_1',
                                              'id_2',
                                              'source',
                                              'mt',
                                              'target',
                                              'source_html_hyperlinks',
                                              'target_htm_hyperlinks'
                                              ])
        df.to_csv(csv_path, index=False)
        return csv_path


if __name__ == '__main__':
    reader = ContentDumpReader()
    count_without_mts = reader.count_without_mts(reader.content_dumps[0])
    print(count_without_mts)
