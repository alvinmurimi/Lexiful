# Lexiful: Advanced Text Matching Tool

Lexiful is a sophisticated text matching and suggestion tool designed for efficient and accurate text processing. It utilizes various natural language processing techniques to provide intelligent matching and highly accurate correction capabilities.

## Features

- TF-IDF vectorization and cosine similarity for precise text matching
- Fuzzy matching algorithms for enhanced accuracy and flexibility
- Context-aware spelling correction with customizable edit distance thresholds
- Comprehensive abbreviation handling and generation capabilities
- Phonetic matching using Soundex and Metaphone algorithms
- N-gram frequency analysis for improved context understanding
- Word embedding support for capturing semantic relationships
- Highly customizable via YAML configuration
- Command-line interface for efficient batch processing
- Web interface for interactive text analysis and matching

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/alvinmurimi/lexiful.git
   cd lexiful
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Configuration

Customize the `config.yaml` file to adjust Lexiful's behavior:

```yaml
input_file: 'descriptions.csv'
csv_description_column: 1
csv_encodings: ['utf-8', 'iso-8859-1', 'windows-1252']
conjunctions: ['and', '&', '+']
ngram_size: 3
embedding_size: 100
window_size: 5
max_edit_distance: 2
model_file: 'lexiful.pkl'
```

## Usage

### Command Line Interface

Use the CLI for quick text matching:

```
python main.py --config config.yaml --input "Your text here" --threshold 70 --matches 5
```

Options:
- `--config`: Path to the configuration file (required)
- `--input`: Input text to match
- `--threshold`: Matching threshold (default: 70)
- `--matches`: Maximum number of matches to return (default: 5)
- `--update`: Path to file containing new descriptions for model update

### Web Interface

Start the Flask web server:

```
python app.py
```

Access the web interface at `http://localhost:5000`.

## Development

### Updating the Model

To update the model with new descriptions:

1. Prepare a file with new descriptions (one per line).
2. Run:
   ```
   python main.py --config config.yaml --update path/to/new_descriptions.txt
   ```

### Extending Functionality

Lexiful is designed to be modular and extensible. Key areas for potential enhancements include:
- Adding new matching algorithms
- Implementing additional pre-processing steps
- Expanding language support
- Optimizing performance for large datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Contact

For questions and support, please open an issue on the GitHub repository.
