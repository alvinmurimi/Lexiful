import argparse
import logging
from lexiful import Lexiful

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the Lexiful application")

    parser = argparse.ArgumentParser(description="Lexiful: Advanced Text Matching Tool")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--input", help="Input text to match")
    parser.add_argument("--threshold", type=float, default=70, help="Matching threshold (default: 70)")
    parser.add_argument("--matches", type=int, default=5, help="Maximum number of matches to return (default: 5)")
    parser.add_argument("--update", help="Path to file containing new descriptions for model update")
    args = parser.parse_args()

    config = Lexiful.load_config(args.config)
    model_file = config['model_file']

    try:
        matcher = Lexiful.load_model(model_file)
        logging.info(f"Loaded existing model from {model_file}")
    except FileNotFoundError:
        logging.info(f"Creating and training new Lexiful model")
        matcher = Lexiful(args.config)
        matcher.save_model(model_file)
        logging.info(f"Model saved to {model_file}")

    if args.update:
        logging.info(f"Updating model with new descriptions from {args.update}")
        with open(args.update, 'r') as f:
            new_descriptions = [line.strip() for line in f if line.strip()]
        matcher.update_model(new_descriptions)
        matcher.save_model(model_file)
        logging.info(f"Model updated and saved to {model_file}")

    if args.input:
        logging.info(f"Processing input: {args.input}")
        matches = matcher.match(args.input, threshold=args.threshold, max_matches=args.matches)

        print(f"\nInput: {args.input}")
        print(f"Matches:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match}")
    else:
        logging.info("No input provided for matching. Use --input to specify input text.")

if __name__ == "__main__":
    main()
