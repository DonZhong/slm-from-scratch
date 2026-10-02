import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    properties = torch.cuda.get_device_properties(0)

    vram_gb = properties.total_memory / (1024 ** 3)

    print("Total VRAM:", round(vram_gb, 2), 'GiB')


if __name__ == "__main__":
    main()