# Getting Started

### Prerequisites

1. **Python 3.9+** installed.
2. Files installed
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root containing:

   ```env
   VOYAGE_API_KEY=<your-voyageai-api-key>
   ```

   Obtain your key at [https://www.voyageai.com/](https://www.voyageai.com/).

---

# 1. Fetch Raw XML Data

The script `new_fetch.py` downloads FRIS data via SOAP. You can choose to fetch **projects** or **publications**.

1. Edit the top of `new_fetch.py`:

   ```python
   PROJECTS = True   # set to False to fetch publications
   ```
2. Run:

   ```bash
   python new_fetch_data.py
   ```
3. Output folders:

   * Projects: `data/rawXml/data_projects_2024_5/`
   * Publications: `data/rawXml/data_publications_2024_5/`

---

# 2. Data Preparation

Ensure your raw XML lives under `data/rawXml/`:

```
data/rawXml/
├── data_projects_2024_5/    
└── data_publications_2024_5/
```

### 2.1 Extract Cleaned CSVs

Use `data_preparation.py` to parse FRIS XML and output cleaned CSVs.

To extract data:

* **For Projects**: Run the function `extractProjectsToCSVFris()` in `data_preparation.py`:

  ```bash
  # Uncomment extractProjectsToCSVFris() if needed
  python data_preparation.py
  ```

  Output: `data/csvs/data_projects_2024_5_FRIS.csv`

* **For Publications**: Run the function `extractPublicationsToCSVFris()` in `data_preparation.py`:

  ```bash
  # Uncomment extractPublicationsToCSVFris() if needed
  python data_preparation.py
  ```

  Output: `data/csvs/data_publications_2024_5_FRIS.csv`

### 2.2 Filter to Publications Linked to Projects

```bash
# Uncomment getWithOnlyProjId() in dataProccesser.py and run
python dataProccesser.py
```

* Output: `data/csvs/data_publications_2024_5_FRIS_WithProjIdsOnly.csv`

### 2.3 Create a Test Sample (Optional)

```bash
# Uncomment createTestSample() in dataProccesser.py and run
python dataProccesser.py
```

* Sample CSV: `data/csvs/data_publications_2024_5_TestSample.csv`

---

# 3. Accuracy Testing

`main.py` has main function for embedding creation and validation.

1. At the top of `main.py`, adjust paths for vector store location, data files, and zip function for text input format.
2. Ensure `.env` is set.
3. Run:

   ```bash
   # uncomment main() function
   python main.py
   ```

Embeddings land in `data/embeddingSaves/` for reuse and output is printed.

---

# 4. Publication Vector Store

Before using `getCorrelatedPublicationData()` in `main.py`, build a publication vector store:

1. In `main.py`, uncomment and call:

   ```python
   createVectorStoreForPublications()
   ```
2. Run:

   ```bash
   python main.py
   ```

* Generates cleaned CSV `data/csvs/data_publications_2024_5_NoDupl.csv`
* Builds Chroma store at `data/vectorStores/data_projects_2024_5_vector_store_voyage_publications`

---

# 5. Running the Web Interface

1. Ensure the publication vector store exists (see step 4).
2. From the main folder, run:

```bash
flask run
```

3. Open `http://127.0.0.1:5000` in your browser.

---

# Project Structure

```
├── data/
│   ├── rawXml/                # Fetched XML
│   ├── csvs/                  # Extracted & sampled CSVs
│   ├── vectorStores/          # Chroma stores
│   └── embeddingSaves/        # Embedding results
├── static/                    # Static files (JS, CSS)
├── templates/                 # HTML templates for Flask
├── data_preparation.py        # XML→CSV & sampling
├── new_fetch.py               # SOAP download script
├── main.py                    # Embedding & tests
├── dataValidation.py          # Test logic
├── Embeddings/                # Embedding wrappers
├── app.py                     # Flask UI
├── requirements.txt           # Dependencies
└── .env                       # Env vars (not in repo)
```

Other files are for tests that failed LDA/BERTopic
