
import json
from urllib.request import urlopen


URL = "http://127.0.0.1:8080/api/health"

EXPECTED = {
    "service": "slm-platform-backend",
    "status": "ok",
}


def main():
    with urlopen(URL, timeout=5) as response:
        assert response.status == 200, (
            f"Unexpected HTTP status: {response.status}"
        )

        data = json.load(response)

    assert data == EXPECTED, (
        f"Unexpected response: {data}"
    )

    print("PASS: Drogon backend health check")


if __name__ == "__main__":
    main()
