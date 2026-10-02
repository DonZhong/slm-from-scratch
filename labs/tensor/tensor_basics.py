import torch

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    a = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    b = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    c = a @ b

    print("device:", device)

    print("a:")

    print(a)

    print("a.shape:", a.shape)

    print("a.dtype:", a.dtype)

    print("a.device:", a.device)

    print("b:")

    print(b)

    print("c = a @ b:")

    print(c)

    expected = torch.tensor(
        [
            [19.0, 22.0],
            [43.0, 50.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    print("result correct:", torch.allclose(c, expected))

    c_cpu = c.cpu()

    print("c on CPU:")
    print(c_cpu)
    print("c_cpu.device:", c_cpu.device)

if __name__ == "__main__":
    main()

    