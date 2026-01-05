"""
Prediction script - generate weather forecasts.
"""
import argparse
import json
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DEFAULT_LOCATION
from inference.predictor import WeatherPredictor, print_forecast

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate weather predictions')
    
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION,
                       help='Location to predict for')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for predictions (JSON)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = WeatherPredictor(
            model_path=args.model,
            location=args.location,
        )
        
        # Generate predictions
        result = predictor.predict()
        
        # Output
        if not args.quiet:
            print_forecast(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Predictions saved to {args.output}")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please train a model first with: python train.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
