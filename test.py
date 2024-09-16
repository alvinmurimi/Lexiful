from lexiful import Lexiful
import csv
from tabulate import tabulate

# Create and train the model
config_file = 'config.yaml'
lexiful_model = Lexiful(config_file)
lexiful_model.save_model('medical_lexiful.pkl')
loaded_model = Lexiful.load_model('medical_lexiful.pkl')

def run_test(title, inputs, max_matches=3):
    print(f"\n## {title}")
    results = []
    for input_text in inputs:
        matches = loaded_model.match(input_text, threshold=60, max_matches=max_matches)
        results.append([input_text, ', '.join(matches) if matches else "No match found"])
    print(tabulate(results, headers=["Input", "Matches"], tablefmt="pipe"))

# Test standard inputs
standard_inputs = [
    "acute myo inf", "COPD", "gastro reflux", "rheumatoid arth",
    "diabetus type 2", "hyper tension", "coronary artery dis",
    "congestive heart failur", "osteo arthritis", "bronchial asthma"
]
run_test("Standard Input Tests", standard_inputs)

# Test abbreviations
abbreviations = ["AMI", "RA", "CAD", "CHF", "OA", "PE"]
run_test("Abbreviation Tests", abbreviations, max_matches=1)

# Test fuzzy matching with typos
typos = [
    "acut myocardial infraction", "gastroesophagal reflux desease",
    "rheumatoid arthritus", "diebetes mellitus", "hipertension"
]
run_test("Fuzzy Matching Tests", typos, max_matches=1)

# Test phonetic matches
phonetic_inputs = [
    "nimonia", "asma", "diubeetees", "athraitis", "hipertenshun"
]
run_test("Phonetic Matching Tests", phonetic_inputs, max_matches=1)
