from src.anomaly_detection import detect_anomalies
from src.data_pipeline import build_analysis_data
from src.generate_docs import generate_documents
from src.modeling import train_models


def main():
    print("1/4 Building account-grain analytical data…")
    build_analysis_data()
    print("2/4 Training and evaluating models…")
    train_models()
    print("3/4 Generating anomaly review queue…")
    detect_anomalies()
    print("4/4 Generating one-page PDFs…")
    generate_documents()
    print("Build complete.")


if __name__ == "__main__":
    main()

