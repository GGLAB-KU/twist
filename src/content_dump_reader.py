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
from openai import OpenAI
from torchmetrics import MeanMetric
from torchmetrics.text import TranslationEditRate
from tqdm import tqdm


class ContentDumpReader:
    def __init__(self, data_dir: str = '../data', num_dumps: int = 2):
        self.data_dir = data_dir
        self.content_dumps = []
        items = list(filter(lambda x: '2024' in x, os.listdir(self.data_dir)))
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

    def _read_dump_as_df_dict(self,
                              dump_alias: str,
                              convert_to_dict_with_id_1: bool = False,
                              keep_na_mt_target: bool = False) -> List[Dict] | Dict[str, List[Dict]]:
        csv_path = self.csv_dict[dump_alias]
        content_df = pd.read_csv(csv_path, dtype={'id_2': pd.StringDtype()})

        if not keep_na_mt_target:
            content_df = content_df.dropna(subset=['mt', 'target'])

        df_records = content_df.to_dict('records')
        df_records.sort(key=lambda x: len(x['target']), reverse=True)

        if convert_to_dict_with_id_1:
            result = defaultdict(list)
            for record in df_records:
                result[record['id_1']] = result[record['id_1']] + [record]
            return result
        else:
            return df_records

    def get_dump_document_with_id(self,
                                  df_records: List[Dict],
                                  doc_id: str) -> List[Dict]:
        dump_sections = list(
            filter(lambda x: doc_id in x['id'],
                   df_records))

        if len(dump_sections) == 0:
            return None
        else:
            return dump_sections

    def get_dump_document_with_id_records_by_doc_id_1(self,
                                                      df_records: Dict[str, List[Dict]],
                                                      doc_id_1: str) -> List[Dict]:
        dump_sections = df_records.get(doc_id_1, [])

        if len(dump_sections) == 0:
            return []
        else:
            return dump_sections

    def compute_ter(self, dump_alias: str) -> float:
        df_records = self._read_dump_as_df_dict(dump_alias)
        df_records = list(filter(lambda x: 5 < len(x['target']) <= 250, df_records))

        return self.compute_ter_from_records(df_records)

    def compute_ter_from_records(self, records) -> float:
        ter = TranslationEditRate().to('cuda')
        current_batch_preds = []
        current_batch_targets = []
        batch_size = 128

        for i, row in enumerate(tqdm(records)):
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

        return self.compute_ned_from_records(df_records)

    def compute_ned_from_records(self, records) -> float:
        metric = MeanMetric()

        for i, row in enumerate(tqdm(records)):
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

        return self.compute_edit_operations_by_type_from_records(df_records)

    def compute_edit_operations_by_type_from_records(self, records):
        delete_metric = MeanMetric()
        equal_metric = MeanMetric()
        insert_metric = MeanMetric()
        replace_metric = MeanMetric()

        for i, row in enumerate(tqdm(records)):
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

        return self.compute_mt_eq_target_from_records(df_records)

    def compute_mt_eq_target_from_records(self, records):
        count = 0
        for i, row in enumerate(tqdm(records)):
            pred = row['mt']
            target = row['target']
            if pred == target:
                count += 1

        return count, len(records)

    def compare_sentence_word_len(self,
                                  dump_alias: str):
        all_df_records = self._read_dump_as_df_dict(dump_alias)

        # all_df_records = list(filter(lambda x: 200 < len(x['target']) <= 250, all_df_records))
        return self.compare_sentence_word_len_from_records(all_df_records)

    def compare_sentence_word_len_from_records(self, records):
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

        all_records_counts = count(records)
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
        search_sentence = sent_tokenize(search_text, 'turkish' if language == 'tr' else 'english')[0].lower()
        dump_section = list(
            filter(lambda x: search_sentence in (x['target'].lower() if language == 'tr' else str(x['source']).lower()),
                   all_df_records))

        if len(dump_section) == 0:
            return None
        elif len(dump_section) == 1:
            found_section = dump_section[0]
            doc_id = found_section['id_1']
            unordered_doc_sections = list(filter(lambda x: x['id_1'] == doc_id, all_df_records))
            return unordered_doc_sections
        else:
            raise ValueError(f'Multiple Dump sections found => {dump_section}')

    def _read_raw_json(self, dump_alias: str) -> Dict:
        dump_folder_path = os.path.join(self.data_dir, dump_alias)
        files = os.listdir(dump_folder_path)
        files = list(filter(lambda f: f.endswith('.json'), files))
        text_json_path = None
        for file in files:
            file_path = os.path.join(dump_folder_path, file)
            if 'text' in file_path:
                text_json_path = file_path

        csv_items = {}

        with open(text_json_path, 'r') as file:
            array_items = ijson.items(file, 'item')
            for item in tqdm(array_items, desc='processing json'):
                item_id = item['id']

                item_source, item_mt, item_target = (item.get('source') or {}).get('content'), (
                        item.get('mt') or {}).get(
                    'content'), (item.get('target') or {}).get('content')

                item_mt_engine = (
                        item.get('mt') or {}).get(
                    'engine')

                item_id_first, item_id_second = item_id.split('/')
                csv_item = [
                    item_id,
                    item_id_first,
                    item_id_second,
                    item_source if item_source != '' else None,
                    item_mt if item_mt != '' else None,
                    item_mt_engine,
                    item_target if item_target != '' else None,
                ]

                csv_items[item_id] = csv_item

        return csv_items

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

        #      "id": "64344/mwAQ",
        #          ...
        #         "mt": {
        #             "engine": "Yandex",

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

    def update_translation_pairs_for_not_found(self):
        file_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs.csv'
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/updated_translation-pairs.csv'
        df = pd.read_csv(file_path)

        dump_alias = reader.content_dumps[1]
        df_records = reader._read_dump_as_df_dict(dump_alias,
                                                  convert_to_dict_with_id_1=True)

        def update_is_done(row):
            if not pd.isna(row['is_done']):
                return row['is_done']

            if row['scientific?'] not in ['TRUE', 'FALSE']:
                return row['is_done']

            doc_id = row['Unnamed: 0']

            result = reader.get_dump_document_with_id_records_by_doc_id_1(df_records,
                                                                          doc_id)

            if result is None:
                return 'NOT_FOUND'
            else:
                return row['is_done']

        df['is_done'] = df.apply(update_is_done, axis=1)

        df.to_csv(updated_path, index=False)

    def end_to_end_annotation_mode(self):
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/updated_translation-pairs.csv'
        translation_pairs_df = pd.read_csv(updated_path)

        dump_alias = reader.content_dumps[1]
        df_records = reader._read_dump_as_df_dict(dump_alias,
                                                  convert_to_dict_with_id_1=True)

        translation_pairs = translation_pairs_df.to_dict('records')

        for translation_pair in tqdm(translation_pairs, desc='annotating...'):
            if translation_pair['is_done'] in ['NOT_FOUND',
                                               'TRUE',
                                               'FALSE']:
                continue

            doc_id = translation_pair['Unnamed: 0']
            result = reader.get_dump_document_with_id_records_by_doc_id_1(df_records,
                                                                          doc_id)

            formatted = [
                f"{item['id']}\t{item['id_1']}\t{item['id_2']}\t\t{item['source']}\t{item['mt']}\t{item['target']}" for
                item in result]

            for f in formatted:
                print(f)
                print('-------------\n')

            while True:
                user_input = input("Press Enter to continue to the next item or type 'Q' and press Enter to quit: ")
                if user_input.upper() == 'Q':
                    return  # Exit the function early
                elif user_input == '':
                    break  # Breaks the inner while loop and continues with the next iteration of the for loop

    ### analysis on scientific vs non-scientific
    def scientific_analysis_count_without_mts(self,
                                              translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        translation_pairs_df = pd.read_csv(translation_pair_path)

        dump_alias = reader.content_dumps[1]
        df_records = reader._read_dump_as_df_dict(dump_alias,
                                                  convert_to_dict_with_id_1=True)

        translation_pairs = translation_pairs_df.to_dict('records')

        total_num_documents = 0
        total_num_sections = 0
        num_docs_without_mts = 0
        num_sections_without_mts = 0  # en az 1 section'ı mt si olanlar arasında
        num_not_founds = 0

        for translation_pair in tqdm(translation_pairs, desc='iterating...'):
            if translation_pair['scientific?'] not in ['TRUE', 'FALSE']:
                continue

            total_num_documents += 1

            doc_id = translation_pair['Unnamed: 0']
            result = reader.get_dump_document_with_id_records_by_doc_id_1(df_records,
                                                                          doc_id)

            # id,id_1,id_2,source,mt,target,source_html_hyperlinks,target_htm_hyperlinks
            if len(result) == 0:
                num_not_founds += 1
            else:
                total_num_sections += len(result)
                current_no_mts = 0
                for record in result:
                    if pd.isna(record['mt']):
                        num_sections_without_mts += 1
                        current_no_mts += 1

                if current_no_mts == len(result):
                    num_docs_without_mts += 1

        return {
            'total_num_documents': total_num_documents,
            'total_num_sections': total_num_sections,
            'num_docs_without_mts': num_docs_without_mts,
            'num_sections_without_mts': num_sections_without_mts,
            'num_not_founds': num_not_founds
        }

    def _scientific_convert_to_records(self,
                                       translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        translation_pairs_df = pd.read_csv(translation_pair_path)
        dump_alias = reader.content_dumps[1]
        df_records = reader._read_dump_as_df_dict(dump_alias,
                                                  convert_to_dict_with_id_1=True)
        translation_pairs = translation_pairs_df.to_dict('records')

        valid_records = []
        for translation_pair in tqdm(translation_pairs, desc='iterating...'):
            if translation_pair['scientific?'] not in ['TRUE', 'FALSE']:
                continue

            doc_id = translation_pair['Unnamed: 0']
            result = reader.get_dump_document_with_id_records_by_doc_id_1(df_records,
                                                                          doc_id)

            # id,id_1,id_2,source,mt,target,source_html_hyperlinks,target_htm_hyperlinks
            if len(result) == 0:
                continue
            else:
                for record in result:
                    valid_records.append({'mt': record['mt'], 'target': record['target']})

        valid_records.sort(key=lambda x: len(x['target']), reverse=True)
        return valid_records

    def scientific_compute_ter(self,
                               translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        valid_records = self._scientific_convert_to_records(translation_pair_path)

        # getting the rid of the instance that has outlier target len
        valid_records = list(filter(lambda x: len(x['target']) <= 250, valid_records))
        return self.compute_ter_from_records(valid_records)

    def scientific_compute_ned(self,
                               translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        valid_records = self._scientific_convert_to_records(translation_pair_path)

        return self.compute_ned_from_records(valid_records)

    def scientific_compute_edit_ops(self,
                                    translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        valid_records = self._scientific_convert_to_records(translation_pair_path)

        return self.compute_edit_operations_by_type_from_records(valid_records)

    def scientific_compute_mt_eq_target(self,
                                        translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        valid_records = self._scientific_convert_to_records(translation_pair_path)

        return self.compute_mt_eq_target_from_records(valid_records)

    def scientific_compare_sentence_word_len(self,
                                             translation_pair_path: str = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/wikimedia-mt-analysis - translation-pairs_final.csv'):
        valid_records = self._scientific_convert_to_records(translation_pair_path)

        return self.compare_sentence_word_len_from_records(valid_records)

    def fill_mt_engine_in_samples(self):
        file_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/scientific_non-sci_samples.csv'
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/updated_scientific_non-sci_samples.csv'
        df = pd.read_csv(file_path)

        dump_alias = reader.content_dumps[1]

        mt_engine_key = 'mt-engine\n(google, yandex etc..)'

        raw_json = self._read_raw_json(dump_alias)

        def update_mt_engine(row):
            doc_id = row['id']

            raw_json_item = raw_json.get(doc_id)
            mt_engine = raw_json_item[5]

            return mt_engine

        df[mt_engine_key] = df.apply(update_mt_engine, axis=1)

        df.to_csv(updated_path, index=False)

    def fill_chatgpt_in_samples(self):
        file_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/scientific_non-sci_samples.csv'
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/updated_scientific_non-sci_samples.csv'
        df = pd.read_csv(file_path)

        # enter your api key
        client = OpenAI(
            api_key="",
        )

        def translate_with_chatgpt(row):
            source_text = row['source']

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system",
                     "content": "You are an expert wikipedian translating articles from English to Turkish."},
                    {"role": "user", "content": f"translate the following from English to Turkish: \"{source_text}\""}
                ]
            )

            chatgpt_text = completion.choices[0].message.content

            return chatgpt_text

        df['chatgpt'] = df.apply(translate_with_chatgpt, axis=1)

        df.to_csv(updated_path, index=False)

    def extract_translation_qualities(self,
                                      ):
        file_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/scientific_non-sci_chatgpt.csv'
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/exported_scientific_non-sci_chatgpt.csv'
        df = pd.read_csv(file_path)

        df_records = df.to_dict('records')

        instances = []

        for record in df_records:
            mt_target_type = record['Type'].lower()
            chatgpt_result = record['chatgpt - result']
            assert 'bad mt' in mt_target_type or 'good mt' in mt_target_type, 'wrong mt val'
            assert 'bad post' in mt_target_type or 'good post' in mt_target_type, 'wrong target val'
            assert 'good' in chatgpt_result or 'bad' in chatgpt_result
            mt_result = True if 'good mt' in mt_target_type else False
            target_result = True if 'good post' in mt_target_type else False
            chatgpt_result = True if 'good' in chatgpt_result else False
            id = record['id']
            is_scientific = True if record['doc type '].lower() == 'scientific' else False

            latin_greek_term = record['latin term\nchanged /\nunchanged']

            if pd.isna(latin_greek_term):
                mt_changed = latin_greek_term
            else:
                mt_changed = 'mt-changed' in latin_greek_term

            mt_engine = record['mt-engine\n(google, yandex etc..)']

            instance = [id,
                        mt_result,
                        target_result,
                        chatgpt_result,
                        is_scientific,
                        mt_changed,
                        mt_engine]
            instances.append(instance)

        df = pd.DataFrame(instances,
                          columns=['id',
                                   'mt',
                                   'target',
                                   'chatgpt',
                                   'is_scientific',
                                   'mt_latin_changed',
                                   'mt_engine'])

        df.to_csv(updated_path, index=False)

    def analyze_translation_quailities(self):
        updated_path = '/home/gsoykan/Desktop/ku/wikimedia-mt-analysis/data/exported_scientific_non-sci_chatgpt.csv'
        data = pd.read_csv(updated_path)
        quality_distribution = data[['mt', 'target', 'chatgpt']].apply(pd.Series.value_counts)
        quality_percentage = quality_distribution / len(data) * 100
        consistent_quality = data[(data['mt'] == data['target']) & (data['target'] == data['chatgpt'])]

        consistent_quality_summary = consistent_quality[['mt', 'target', 'chatgpt']].apply(pd.Series.value_counts)

        return quality_distribution, quality_percentage, consistent_quality_summary, quality_rates, quality_by_type


if __name__ == '__main__':
    reader = ContentDumpReader(num_dumps=2)
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

    # pair_doc_sections = reader.find_dump_entry_of_wiki_article(reader.content_dumps[0],
    #                                                           "Elektron",
    #                                                           language='tr')
    # print(pair_doc_sections)

    # dump_alias = reader.content_dumps[1]
    # df_records = reader._read_dump_as_df_dict(dump_alias,
    #                                           convert_to_dict_with_id_1=True)
    # doc_id = 156853
    # result = reader.get_dump_document_with_id(df_records, doc_id)
    # print(result)
    #
    # reader.update_translation_pairs_for_not_found()

    # reader.end_to_end_annotation_mode()

    #### scientific values

    # scientific_counts = reader.scientific_analysis_count_without_mts()
    # print(scientific_counts)

    # ter_value = reader.scientific_compute_ter()
    # print(ter_value)

    # ned_value = reader.scientific_compute_ned()
    # print(ned_value)
    # mean_ops_by_type = reader.scientific_compute_edit_ops()
    # print(mean_ops_by_type)

    # mt_eq_target = reader.scientific_compute_mt_eq_target()
    # print(mt_eq_target)

    # stats = reader.scientific_compare_sentence_word_len()
    # print(stats)

    # reader.fill_mt_engine_in_samples()

    # reader.fill_chatgpt_in_samples()

    # reader.extract_translation_qualities()

    reader.analyze_translation_quailities()
