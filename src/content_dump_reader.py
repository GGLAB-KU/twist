import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import Levenshtein
import ijson
import numpy as np
import pandas as pd
import wikipediaapi
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from torchmetrics import MeanMetric
from torchmetrics.text import TranslationEditRate
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

    def _read_dump_as_df_dict(self, dump_alias: str) -> List[Dict]:
        csv_path = self.csv_dict[dump_alias]
        content_df = pd.read_csv(csv_path, dtype={'id_2': pd.StringDtype()})
        content_df = content_df.dropna(subset=['mt', 'target'])
        df_records = content_df.to_dict('records')
        df_records.sort(key=lambda x: len(x['target']), reverse=True)
        return df_records

    def compute_ter(self, dump_alias: str) -> float:
        df_records = self._read_dump_as_df_dict(dump_alias)
        df_records = list(filter(lambda x: 5 < len(x['target']) <= 250, df_records))

        ter = TranslationEditRate().to('cuda')
        current_batch_preds = []
        current_batch_targets = []
        batch_size = 128

        for i, row in enumerate(tqdm(df_records)):
            pred = row['mt']
            target = row['target']
            current_batch_preds.append(pred)
            current_batch_targets.append([target])
            if i != 0 and i % batch_size == 0:
                ter.update(current_batch_preds, current_batch_targets)
                current_batch_preds = []
                current_batch_targets = []

        result = ter.compute()
        return result

    def compute_ned(self, dump_alias: str, filter_by_len: bool = True) -> float:
        df_records = self._read_dump_as_df_dict(dump_alias)
        if filter_by_len:
            df_records = list(filter(lambda x: 5 < len(x['target']) <= 250, df_records))

        metric = MeanMetric()

        for i, row in enumerate(tqdm(df_records)):
            pred = row['mt']
            target = row['target']
            edit_distance = Levenshtein.distance(pred, target)
            normalized_edit_distance = edit_distance / max(len(pred), len(target))
            metric.update(normalized_edit_distance)

        result = metric.compute()
        return result

    def compute_edit_operations_by_type(self,
                                        dump_alias: str,
                                        filter_by_len: bool = True) -> float:
        df_records = self._read_dump_as_df_dict(dump_alias)
        if filter_by_len:
            df_records = list(filter(lambda x: 5 < len(x['target']) <= 250, df_records))

        delete_metric = MeanMetric()
        equal_metric = MeanMetric()
        insert_metric = MeanMetric()
        replace_metric = MeanMetric()

        for i, row in enumerate(tqdm(df_records)):
            pred = row['mt']
            target = row['target']
            opcodes = Levenshtein.opcodes(pred, target)
            op_dict = defaultdict(lambda: 0)
            for opcode in opcodes:
                op_type = opcode[0]
                if op_type != 'delete':
                    cost = abs(opcode[4] - opcode[3])
                else:
                    cost = abs(opcode[2] - opcode[1])
                op_dict[op_type] = op_dict[op_type] + cost

            total_ops = 0
            for k, v in op_dict.items():
                total_ops += v

            normalized_op_dict = {k: v / total_ops for k, v in op_dict.items()}
            delete_metric.update(normalized_op_dict.get('delete', 0))
            equal_metric.update(normalized_op_dict.get('equal', 0))
            insert_metric.update(normalized_op_dict.get('insert', 0))
            replace_metric.update(normalized_op_dict.get('replace', 0))

        delete_result = delete_metric.compute()
        equal_result = equal_metric.compute()
        insert_result = insert_metric.compute()
        replace_result = replace_metric.compute()
        return delete_result, equal_result, insert_result, replace_result

    def compute_mt_eq_target(self, dump_alias: str, filter_by_len: bool = True) -> Tuple[int, int]:
        df_records = self._read_dump_as_df_dict(dump_alias)

        if filter_by_len:
            df_records = list(filter(lambda x: 5 < len(x['target']) <= 250, df_records))

        count = 0
        for i, row in enumerate(tqdm(df_records)):
            pred = row['mt']
            target = row['target']
            if pred == target:
                count += 1

        return count, len(df_records)

    def compare_sentence_word_len(self,
                                  dump_alias: str):
        all_df_records = self._read_dump_as_df_dict(dump_alias)

        # all_df_records = list(filter(lambda x: 200 < len(x['target']) <= 250, all_df_records))

        def count(df_records):
            pred_sent_lens = []
            target_sent_lens = []
            pred_token_lens = []
            target_token_lens = []

            for i, row in enumerate(tqdm(df_records)):
                pred = row['mt']
                target = row['target']

                pred_sents = sent_tokenize(pred, 'turkish')
                target_sents = sent_tokenize(target, 'turkish')

                pred_words = word_tokenize(pred, 'turkish')
                target_words = word_tokenize(target, 'turkish')

                pred_sent_lens.append(len(pred_sents))
                target_sent_lens.append(len(target_sents))

                pred_token_lens.append(len(pred_words))
                target_token_lens.append(len(target_words))

            return (pred_sent_lens,
                    target_sent_lens,
                    pred_token_lens,
                    target_token_lens)

        def compute_stats(counts,
                          target_word_len_range: Optional[Tuple[int, int]] = None):
            if target_word_len_range is None:
                return (np.array(counts[0]).mean(),
                        np.array(counts[1]).mean(),
                        np.array(counts[2]).mean(),
                        np.array(counts[3]).mean())
            else:
                target_word_len = np.array(counts[3])
                range_min, range_max = target_word_len_range
                ind = np.multiply(range_min < target_word_len, target_word_len <= range_max)
                return (np.array(counts[0])[ind].mean(),
                        np.array(counts[1])[ind].mean(),
                        np.array(counts[2])[ind].mean(),
                        np.array(counts[3])[ind].mean())

        all_records_counts = count(all_df_records)
        pred_sent_lens, target_sent_lens, _, target_token_lens = all_records_counts

        all_records_stats = compute_stats(all_records_counts, target_word_len_range=None)

        records_stats_250 = compute_stats(all_records_counts, target_word_len_range=(5, 250))

        bucket_stats = {}

        uniq_token_lengths, uniq_token_counts = np.unique(target_token_lens, return_counts=True)

        #  Interquartile Range (IQR) outlier removal
        Q1 = np.percentile(uniq_token_lengths, 25)
        Q3 = np.percentile(uniq_token_lengths, 75)
        IQR = Q3 - Q1
        upper_word_len_bound = int(Q3 + 1.5 * IQR)  # 2197

        step_size = int((upper_word_len_bound - 3) / 100)  # 21

        for i in tqdm(range(3, upper_word_len_bound, step_size)):
            stats = compute_stats(all_records_counts, target_word_len_range=(i, i + step_size))
            bucket_stats[f'{str(i)} - {str(i + step_size)}'] = stats
        bucket_stats = {k: v for (k, v) in bucket_stats.items() if not pd.isna(v[0])}

        return all_records_stats, records_stats_250, bucket_stats, (pred_sent_lens, target_sent_lens)

    def find_dump_entry_of_wiki_article(self,
                                        dump_alias: str,
                                        wiki_title: str,
                                        language: str = 'tr') -> Optional[List[Dict]]:
        all_df_records = self._read_dump_as_df_dict(dump_alias)
        wiki = wikipediaapi.Wikipedia('GGWikimedia (grkn245@gmail.com)', language)
        wiki_page = wiki.page(wiki_title, )
        assert wiki_page.exists(), f'Wikipedia page {wiki_title} does not exist'

        # print(wiki_page.text)  # full text
        # print(wiki_page.sections)  # sections

        def get_first_lowest_section_text(page) -> str:
            if len(page.sections) != 0:
                return get_first_lowest_section_text(page.sections[0])
            else:
                return page.text

        search_text = get_first_lowest_section_text(wiki_page)
        search_sentence = sent_tokenize(search_text, 'turkish')[0]
        dump_section = list(
            filter(lambda x: search_sentence in x['target'], all_df_records))

        if len(dump_section) == 0:
            return None
        elif len(dump_section) == 1:
            found_section = dump_section[0]
            doc_id = found_section['id_1']
            unordered_doc_sections = list(filter(lambda x: x['id_1'] == doc_id, all_df_records))
            return unordered_doc_sections
        else:
            raise ValueError(f'Multiple Dump sections found => {dump_section}')

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
    # count_without_mts = reader.count_without_mts(reader.content_dumps[0])
    # print(count_without_mts)
    # ter_value = reader.compute_ter(reader.content_dumps[0])
    # print(ter_value)

    # ned_value = reader.compute_ned(reader.content_dumps[0], filter_by_len=False)
    # print(ned_value)
    # mean_ops_by_type = reader.compute_edit_operations_by_type(reader.content_dumps[0], filter_by_len=True)
    # print(mean_ops_by_type)

    # mt_eq_target_filtered = reader.compute_mt_eq_target(reader.content_dumps[0], filter_by_len=False)
    # mt_eq_target = reader.compute_mt_eq_target(reader.content_dumps[0], filter_by_len=True)
    # print(mt_eq_target, mt_eq_target_filtered)

    # stats = reader.compare_sentence_word_len(reader.content_dumps[0])
    # print(stats)

    pair_doc_sections = reader.find_dump_entry_of_wiki_article(reader.content_dumps[0],
                                                               "Elektron dizilimi",
                                                               language='tr')
    print(pair_doc_sections)
