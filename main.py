"""
Main CLI for the Climate Predictor project.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Climate Predictor - LSTM-based weather prediction',
        usage='python main.py <command> [options]'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch historical weather data')
    fetch_parser.add_argument('--location', type=str, default='berlin',
                             help='Location (berlin, munich, pune)')
    fetch_parser.add_argument('--years', type=int, default=10,
                             help='Number of years to fetch')
    fetch_parser.add_argument('--start-year', type=int, default=None,
                             help='Start year (default: current year - years)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--location', type=str, default='berlin')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--location', type=str, default='berlin')
    predict_parser.add_argument('--output', type=str, default=None)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show project info')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        from config.settings import DATA_START_YEAR, DATA_END_YEAR
        from data.fetcher import WeatherDataFetcher
        
        start_year = args.start_year or (DATA_END_YEAR - args.years + 1)
        end_year = DATA_END_YEAR
        
        print(f"Fetching {args.location} data from {start_year} to {end_year}...")
        fetcher = WeatherDataFetcher(location=args.location)
        df = fetcher.fetch_years(start_year, end_year)
        print(f"Done! {len(df)} hourly records saved.")
        
    elif args.command == 'train':
        import subprocess
        cmd = [
            sys.executable, 'train.py',
            '--location', args.location,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
        ]
        subprocess.run(cmd)
        
    elif args.command == 'predict':
        from inference.predictor import WeatherPredictor, print_forecast
        
        predictor = WeatherPredictor(location=args.location)
        result = predictor.predict()
        print_forecast(result)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")
            
    elif args.command == 'info':
        from config.settings import LOCATIONS, HOURLY_VARIABLES, TARGET_VARIABLES
        
        print("\n=== Climate Predictor ===")
        print("\nAvailable Locations:")
        for key, loc in LOCATIONS.items():
            print(f"  - {key}: {loc.name} ({loc.latitude}, {loc.longitude})")
        
        print(f"\nWeather Features: {len(HOURLY_VARIABLES)}")
        print(f"Target Variables: {TARGET_VARIABLES}")
        print("\nCommands:")
        print("  python main.py fetch --location berlin --years 10")
        print("  python main.py train --epochs 50")
        print("  python main.py predict")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
