# Lexiful 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Lexiful is a powerful, lightweight natural language processing engine designed for high-precision text matching, intelligent suggestion, and advanced correction capabilities. By leveraging cutting-edge NLP techniques, Lexiful provides unparalleled accuracy and flexibility in text processing tasks, particularly in industry-specific scenarios.

## 🚀 Features

- **🎯 Intelligent Text Matching**: Utilizes TF-IDF vectorization and cosine similarity for precise matching results.
- **🔍 Fuzzy Matching Algorithms**: Implements advanced fuzzy matching for enhanced accuracy and flexibility.
- **✏️ Context-Aware Spelling Correction**: Offers sophisticated spelling correction with customizable edit distance thresholds.
- **📚 Comprehensive Abbreviation Handling**: Generates and processes abbreviations intelligently.
- **🔊 Phonetic Matching**: Employs Soundex and Metaphone algorithms for sound-based text matching.
- **📊 N-gram Frequency Analysis**: Enhances context understanding through n-gram analysis.
- **🧬 Word Embedding Support**: Captures semantic relationships using word embeddings.
- **⚙️ Highly Customizable**: Configurable via YAML for tailored performance.
- **🔄 Adaptive Learning**: Supports model updates and user-defined corrections for continuous improvement.

## 🏭 Industry-Specific Applications

Lexiful is engineered as a robust solution for industry-specific scenarios where matching user input against predefined data is crucial. It excels in:

- **🎯 Targeted Matching**: Optimized for specific industry terminologies and data structures.
- **🔒 Data Consistency**: Reduces free-type errors by matching user input to standardized entries.
- **⚡ Efficiency**: Faster and more resource-efficient than broad AI models for specific matching tasks.
- **🛠️ Customizability**: Easily adaptable to various industries and specific organizational needs.
- **🔐 Privacy-Focused**: Operates on local, predefined datasets without relying on external knowledge bases.

## 🛠️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/alvinmurimi/lexiful.git
    cd lexiful
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download NLTK data:
    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```

## ⚙️ Configuration

Customize the `config.yaml` file to adjust Lexiful's behavior:

```yaml
input_file: 'text.txt'
csv_description_column: 1
csv_encodings: ['utf-8', 'iso-8859-1', 'windows-1252']
conjunctions: ['and', '&', '+', '/']
fuzzy_match_algorithm: 'token_set_ratio'
ngram_size: 3
embedding_size: 100
window_size: 5
max_edit_distance: 2
model_file: 'model.pkl'
```

## 📖 Usage

### Basic Usage

```python
from lexiful import Lexiful

# Initialize Lexiful
lexiful = Lexiful('config.yaml')

# Match input text
matches = lexiful.match("Your input text", threshold=60, max_matches=5)
print(matches)
```

### Advanced Usage and Model Improvement

#### User Corrections
```python
lexiful.learn_correction("original_word", "corrected_word")
```

#### Model Updates
```python
new_descriptions = ["New description 1", "New description 2"]
lexiful.update_model(new_descriptions)
```

#### Save and Load Model
```python
# Save model
lexiful.save_model("model.pkl")

# Load model
loaded_lexiful = Lexiful.load_model("model.pkl")
```

## 🧪 Testing

We use `test.py` to evaluate our model's performance on medical terminology. The model is trained on data from `descriptions.csv`, which contains 11 medical terms.

### Test Categories

- **Standard Inputs**: Tests partial terms and common medical phrases.
- **Abbreviation**: Checks recognition of medical acronyms.
- **Fuzzy Matching**: Evaluates handling of misspellings and typos.
- **Phonetic Matching**: Tests ability to match phonetically similar inputs.

Below are the test results:
```bash
## Standard Input Tests
| Input                   | Matches                               |
|:------------------------|:--------------------------------------|
| acute myo inf           | Acute Myocardial Infarction           |
| COPD                    | Chronic Obstructive Pulmonary Disease |
| gastro reflux           | Gastroesophageal Reflux Disease       |
| rheumatoid arth         | Rheumatoid Arthritis                  |
| diabetus type 2         | Diabetes Mellitus Type 2              |
| hyper tension           | Hypertension                          |
| coronary artery dis     | Coronary Artery Disease               |
| congestive heart failur | Congestive Heart Failure              |
| osteo arthritis         | Osteoarthritis, Rheumatoid Arthritis  |
| bronchial asthma        | Asthma                                |

## Abbreviation Tests
| Input   | Matches                     |
|:--------|:----------------------------|
| AMI     | Acute Myocardial Infarction |
| RA      | Rheumatoid Arthritis        |
| CAD     | Coronary Artery Disease     |
| CHF     | Congestive Heart Failure    |
| OA      | Osteoarthritis              |

## Fuzzy Matching Tests
| Input                          | Matches                         |
|:-------------------------------|:--------------------------------|
| acut myocardial infraction     | Acute Myocardial Infarction     |
| gastroesophagal reflux desease | Gastroesophageal Reflux Disease |
| rheumatoid arthritus           | Rheumatoid Arthritis            |
| diebetes mellitus              | Diabetes Mellitus Type 2        |
| hipertension                   | Hypertension                    |

## Phonetic Matching Tests
| Input        | Matches                  |
|:-------------|:-------------------------|
| nimonia      | Pneumonia                |
| asma         | Asthma                   |
| dayabites    | Diabetes Mellitus Type 2 |
| athraitis    | Osteoarthritis           |
| hipertenshun | Hypertension             |
```


## 🚀 Development

### Extending Functionality

Lexiful provides a solid starting point for text matching and entity recognition. Key areas for potential enhancements include:

- Implementing more sophisticated pre-processing steps in the `preprocess` method
- Adding new matching algorithms to the `match` method
- Expanding language support by incorporating multilingual resources
- Optimizing performance for large datasets through efficient data structures
- Fully integrating word embeddings into the matching process

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions or feedback, please open an issue or contact [Alvin Mayende](mailto:alvinmayende@gmail.com)

---