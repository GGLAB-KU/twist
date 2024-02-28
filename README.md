# Setup Instructions

### Creating the Conda Environment and Installing Dependencies

1. **Create a Conda environment** by running the following command:
   ```
   conda create --name <env_name> python=3.10
   ```
   Replace `<env_name>` with your desired environment name.

2. **Activate the Conda environment**:
   ```
   conda activate <env_name>
   ```

3. **Install the required dependencies** using the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

### How to Use?

#### ContentDumpDownloader

This tool is used to download content dumps from Wikipedia that include translations from English to Turkish and contain both HTML and text (JSON) data. After downloading, the dumps are uncompressed. The HTML files are approximately 6-7 GB, and the text files are around 500 MB. The downloaded files are saved in the 'data' folder in the project root.

**Usage Example**:
```python
dump_downloader = ContentDumpDownloader(num_latest_dumps_to_use=2)
dump_downloader.download_dumps()
```

#### ContentDumpReader

This tool reads the downloaded JSON files and merges HTML and text data into a CSV file. The CSV rows will look like the following:

| id         | id_1  | id_2 | Source                                                                                                            | mt | Target                                                                                                            | source_html_hyperlinks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | target_html_hyperlinks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|------------|-------|------|-------------------------------------------------------------------------------------------------------------------|----|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 40287/mw0Q | 40287 | mw0Q | External links                                                                                                    |    | External Links                                                                                                    | [], []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | [], []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 40287/mw0g | 40287 | mw0g | American Veterinary Medical Association European Board of Veterinary Specialisation NetVet Vet Students Abroad    |    | American Veterinary Medical Association European Board of Veterinary Specialisation NetVet Vet Students Abroad    | [{"href": "http://www.avma.org", "id": "mw1A", "data_linkid": "530", "rel": ["mw:ExtLink"], "string": "American Veterinary Medical Association", "text": "American Veterinary Medical Association"}, {"href": "http://www.ebvs.be/", "id": "mw1g", "data_linkid": "533", "rel": ["mw:ExtLink"], "string": "European Board of Veterinary Specialisation", "text": "European Board of Veterinary Specialisation"}, {"href": "http://netvet.wustl.edu/vspecial.htm", "id": "mw2A", "data_linkid": "536", "rel": ["mw:ExtLink"], "string": "NetVet", "text": "NetVet"}, {"href": "http://vetstudents.net", "id": "mw2g", "data_linkid": "539", "rel": ["mw:ExtLink"], "string": "Vet Students Abroad", "text": "Vet Students Abroad"}] | [{"href": "http://www.avma.org", "id": "mw1A", "data_linkid": "530", "rel": ["mw:ExtLink"], "string": "American Veterinary Medical Association", "text": "American Veterinary Medical Association"}, {"href": "http://www.ebvs.be/", "id": "mw1g", "data_linkid": "533", "rel": ["mw:ExtLink"], "string": "European Board of Veterinary Specialisation", "text": "European Board of Veterinary Specialisation"}, {"href": "http://netvet.wustl.edu/vspecial.htm", "id": "mw2A", "data_linkid": "536", "rel": ["mw:ExtLink"], "string": "NetVet", "text": "NetVet"}, {"href": "http://vetstudents.net", "id": "mw2g", "data_linkid": "539", "rel": ["mw:ExtLink"], "string": "Vet Students Abroad", "text": "Vet Students Abroad"}] |


**Usage Example**:
```python
reader = ContentDumpReader(num_dumps=1)
count_without_mts = reader.count_without_mts(reader.content_dumps[0])

ter_value = reader.compute_ter(reader.content_dumps[0])

ned_value = reader.compute_ned(reader.content_dumps[0], filter_by_len=False)

mt_eq_target_filtered = reader.compute_mt_eq_target(reader.content_dumps[0], filter_by_len=False)
mt_eq_target = reader.compute_mt_eq_target(reader.content_dumps[0], filter_by_len=True)

stats = reader.compare_sentence_word_len(reader.content_dumps[0])
```

### Useful Links

- **Roadmap**: [View the roadmap here](https://docs.google.com/document/d/1m-xCAAIDmP6fSCcrJJ-BF8ICF0kjCaDXBXZuy6wD_e4/edit?usp=sharing)
- **Main Google Sheets for Task Tracking and Results**: [Access the spreadsheet here](https://docs.google.com/spreadsheets/d/1iNXDL1k2kcV5KXFPXtKLKKfIRK80ZU-HS-iSFRhIIdw/edit?usp=sharing)
- **Wikipedia Content Dump**: [Download content dumps here](https://dumps.wikimedia.org/other/contenttranslation)