"""
Script to download the GPT4All Falcon model
"""
from models import get_model_path, download_model, get_model_info


def main():
    # Get Falcon model info
    model_info = get_model_info("GPT4All Falcon")
    if model_info:
        print(f"Downloading {model_info.name}...")
        model_path = download_model(model_info)
        if model_path:
            print(f"\nSuccessfully downloaded {model_info.name} to {model_path}")
            print("\nRecommended use cases:")
            for use_case in model_info.recommended_for:
                print(f"- {use_case}")
        else:
            print(f"\nFailed to download {model_info.name}")
    else:
        print("Model information not found")

if __name__ == "__main__":
    main()
