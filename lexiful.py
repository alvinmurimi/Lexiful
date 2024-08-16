import re
import csv
import os
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
from collections import Counter
from jellyfish import soundex, metaphone
from nltk.util import ngrams
import logging
from typing import List, Dict, Tuple
import yaml
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('stopwords', quiet=True)

class Lexiful:
    def __init__(self, config_file: str):
        # Initialize core components
        self.config = self.load_config(config_file)
        self.descriptions = self.load_descriptions(self.config['input_file'])
        self.stop_words = set(stopwords.words('english'))
        self.preprocessed_descriptions = [self.preprocess(desc) for desc in self.descriptions]
        self.abbreviations = self.generate_abbreviations()
        self.word_embeddings = self.train_word_embeddings()
        self.word_freq = self.build_word_frequency()
        self.ngram_freq = self.build_ngram_frequency(self.config['ngram_size'])
        self.phonetic_map = self.build_phonetic_map()
        self.user_corrections = {}

    @staticmethod
    def load_config(config_file: str) -> Dict:
        # Load YAML configuration file
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_descriptions(self, input_file: str) -> List[str]:
        descriptions = []
        encodings = self.config['csv_encodings']
        
        file_extension = os.path.splitext(input_file)[1].lower()

        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as file:
                    if file_extension == '.csv':
                        # Process CSV file
                        csv_reader = csv.reader(file)
                        next(csv_reader)  # Skip header row
                        desc_column = self.config['csv_description_column']
                        for row in csv_reader:
                            if len(row) > desc_column:
                                descriptions.append(row[desc_column].strip())
                    elif file_extension in ['.txt', '.text']:
                        # Process plain text file
                        descriptions = [line.strip() for line in file if line.strip()]
                    else:
                        # Attempt to detect CSV or treat as plain text
                        file.seek(0)
                        dialect = csv.Sniffer().sniff(file.read(1024))
                        file.seek(0)
                        if dialect:
                            csv_reader = csv.reader(file, dialect)
                            next(csv_reader)  # Skip header row
                            desc_column = self.config['csv_description_column']
                            for row in csv_reader:
                                if len(row) > desc_column:
                                    descriptions.append(row[desc_column].strip())
                        else:
                            file.seek(0)
                            descriptions = [line.strip() for line in file if line.strip()]
                    
                logging.info(f"Successfully loaded {len(descriptions)} descriptions from {file_extension} file.")
                return descriptions
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.error(f"An error occurred while reading the file: {str(e)}")
                return []
        
        logging.error(f"Failed to read the file with any of the attempted encodings: {encodings}")
        return []

    def preprocess(self, text: str) -> str:
        # Remove stopwords and convert to lowercase
        return ' '.join([word.lower() for word in simple_preprocess(text) if word not in self.stop_words])

    def generate_abbreviations(self) -> Dict[str, List[str]]:
        abbr_dict = {}
        conjunctions = self.config['conjunctions']
        
        for desc in self.descriptions:
            words = desc.split()
            
            # Generate standard abbreviation (e.g., "ABC" for "Alpha Beta Company")
            if len(words) > 1:
                abbr = ''.join(word[0].upper() for word in words)
                if abbr not in abbr_dict:
                    abbr_dict[abbr] = []
                abbr_dict[abbr].append(desc)
            
            # Generate abbreviations with conjunctions
            for conj in conjunctions:
                if conj in words:
                    conj_indices = [i for i, word in enumerate(words) if word == conj]
                    for idx in conj_indices:
                        # Full abbreviation with conjunction (e.g., "A&B" for "Alpha and Beta")
                        abbr_full = ''.join(word[0].upper() for word in words[:idx]) + conj + ''.join(word[0].upper() for word in words[idx+1:])
                        if abbr_full not in abbr_dict:
                            abbr_dict[abbr_full] = []
                        abbr_dict[abbr_full].append(desc)
                        
                        # Short abbreviation without conjunction (e.g., "AB" for "Alpha and Beta")
                        abbr_short = ''.join(word[0].upper() for word in words if word != conj)
                        if abbr_short not in abbr_dict:
                            abbr_dict[abbr_short] = []
                        abbr_dict[abbr_short].append(desc)
                        
                        # Two-letter abbreviation with conjunction (e.g., "A&B" for "Alpha and Beta Company")
                        if idx > 0 and idx < len(words) - 1:
                            abbr_two = words[idx-1][0].upper() + conj + words[idx+1][0].upper()
                            if abbr_two not in abbr_dict:
                                abbr_dict[abbr_two] = []
                            abbr_dict[abbr_two].append(desc)
            
            # Handle 'of' separately (e.g., "DOJ" for "Department of Justice")
            if 'of' in words:
                of_index = words.index('of')
                abbr_of = ''.join(word[0].upper() for word in words if words.index(word) != of_index)
                if abbr_of not in abbr_dict:
                    abbr_dict[abbr_of] = []
                abbr_dict[abbr_of].append(desc)

        return abbr_dict

    def train_word_embeddings(self):
        # Train Word2Vec model on preprocessed descriptions
        sentences = [simple_preprocess(desc) for desc in self.descriptions]
        model = Word2Vec(sentences, vector_size=self.config['embedding_size'], window=self.config['window_size'], min_count=1, workers=4)
        return model.wv

    def build_word_frequency(self) -> Counter:
        # Count word occurrences across all descriptions
        words = [word for desc in self.descriptions for word in simple_preprocess(desc)]
        return Counter(words)

    def build_ngram_frequency(self, n: int) -> Counter:
        # Build n-gram frequency distribution
        ngram_freq = Counter()
        for desc in self.descriptions:
            words = simple_preprocess(desc)
            ngram_freq.update(ngrams(words, n))
        return ngram_freq

    def build_phonetic_map(self) -> Dict[str, List[str]]:
        # Create phonetic mapping for efficient similarity search
        phonetic_map = {}
        for word in self.word_freq:
            soundex_code = soundex(word)
            metaphone_code = metaphone(word)
            if soundex_code not in phonetic_map:
                phonetic_map[soundex_code] = []
            if metaphone_code not in phonetic_map:
                phonetic_map[metaphone_code] = []
            phonetic_map[soundex_code].append(word)
            phonetic_map[metaphone_code].append(word)
        return phonetic_map

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        # Calculate Levenshtein distance between two strings
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def find_closest_word(self, word: str, context: Tuple[str, str] = None) -> str:
        # Find the closest known word based on edit distance and context
        if word in self.user_corrections:
            return self.user_corrections[word]

        min_distance = float('inf')
        closest_word = word

        soundex_code = soundex(word)
        metaphone_code = metaphone(word)
        phonetic_matches = set(self.phonetic_map.get(soundex_code, []) + self.phonetic_map.get(metaphone_code, []))

        for similar_word in phonetic_matches:
            distance = self.levenshtein_distance(word, similar_word)
            if distance < min_distance:
                min_distance = distance
                closest_word = similar_word

        if closest_word == word:
            for known_word in self.word_freq:
                distance = self.levenshtein_distance(word, known_word)
                if distance < min_distance:
                    min_distance = distance
                    closest_word = known_word

        if context and closest_word == word:
            for known_word in self.word_freq:
                if self.ngram_freq.get((context[0], known_word, context[1]), 0) > self.ngram_freq.get((context[0], closest_word, context[1]), 0):
                    closest_word = known_word

        return closest_word if min_distance <= self.config['max_edit_distance'] else word

    def correct_input(self, input_text: str) -> str:
        # Apply spelling correction to input text
        words = simple_preprocess(input_text)
        corrected_words = []
        for i, word in enumerate(words):
            context = (words[i-1] if i > 0 else '', words[i+1] if i < len(words)-1 else '')
            corrected_words.append(self.find_closest_word(word, context))
        return ' '.join(corrected_words)

    def match(self, input_text: str, threshold: float = 70, max_matches: int = 5) -> List[str]:
        # Remove non-alphabetic characters and convert to uppercase
        processed_input = re.sub(r'[^a-zA-Z&]', '', input_text.upper())
        
        # Check if input is an abbreviation
        if processed_input in self.abbreviations:
            return self.abbreviations[processed_input][:max_matches]
        
        preprocessed_input = self.preprocess(input_text)
        input_words = preprocessed_input.split()
        
        # Apply spelling correction
        corrected_input = self.correct_input(preprocessed_input)
        
        matches = []
        for i, desc in enumerate(self.preprocessed_descriptions):
            # Calculate fuzzy match ratio
            ratio = fuzz.partial_ratio(corrected_input, desc)
            
            desc_words = desc.split()
            if set(input_words) & set(desc_words):
                # Calculate embedding similarity
                input_vec = np.mean([self.word_embeddings[w] for w in input_words if w in self.word_embeddings], axis=0)
                desc_vec = np.mean([self.word_embeddings[w] for w in desc_words if w in self.word_embeddings], axis=0)
                if input_vec.size and desc_vec.size:
                    embedding_sim = np.dot(input_vec, desc_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(desc_vec))
                    ratio = max(ratio, embedding_sim * 100)
            
            # Calculate edit distance ratio
            edit_dist = self.levenshtein_distance(corrected_input, desc)
            max_len = max(len(corrected_input), len(desc))
            edit_ratio = (max_len - edit_dist) / max_len * 100 if max_len > 0 else 0
            
            final_ratio = max(ratio, edit_ratio)
            
            if final_ratio >= threshold:
                matches.append((self.descriptions[i], final_ratio))
        
        # Sort matches by ratio and return top results
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:max_matches]]

    def learn_correction(self, original_word: str, corrected_word: str):
        # Add user-defined correction to the dictionary
        self.user_corrections[original_word] = corrected_word

    def save_model(self, filename: str):
        # Serialize and save the model to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename: str) -> 'Lexiful':
        # Load a serialized model from a file
        with open(filename, 'rb') as f:
            return pickle.load(f)
