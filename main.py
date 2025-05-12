import argparse
from experiments.experiment_runner import run_experiments
from utils.metrics import compare_models

def main():
    parser = argparse.ArgumentParser(description="Akıllı Market Stok Optimizasyonu Aracı")
    parser.add_argument("--mode", choices=["run", "plot"], required=True,
                        help="'run': modelleri çalıştırır, 'plot': grafik üretir")
    parser.add_argument("--data", default="data/demand_data.csv", help="CSV veri dosyası yolu")

    args = parser.parse_args()

    if args.mode == "run":
        run_experiments(data_path=args.data)
    elif args.mode == "plot":
        compare_models("reports/lp_results.csv", "reports/ga_results.csv")

if __name__ == "__main__":
    main()
