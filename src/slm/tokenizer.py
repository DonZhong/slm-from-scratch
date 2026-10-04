class CharacterTokenizer:
    def __init__(
        self,
        vocabulary: list[str],
        unk_token: str = "<unk>",
    ):

        if not vocabulary:
            raise ValueError("vocabulary cannot be empty")

        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("vocabulary cannot contain duplicate tokens")

        if any(len(token) != 1 for token in vocabulary):
            raise ValueError(
                "vocabulary must contain single-character tokens"
            )

        self.unk_token = unk_token

        tokens = [unk_token, *vocabulary]

        self.token_to_id = {
            token: token_id
            for token_id, token in enumerate(tokens)
        }

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    @classmethod
    def from_text(cls, text: str):
        if not text:
            raise ValueError("text cannot be empty")

        vocabulary = sorted(set(text))

        return cls(vocabulary)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, text: str) -> list[int]:
        return[
            self.token_to_id.get(character, self.unk_id)
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        tokens = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(
                    f"token_id {token_id} is not in the vocabulary"
                )

            tokens.append(self.id_to_token[token_id])

        return "".join(tokens)