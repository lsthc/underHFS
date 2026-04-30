from underhfs.nn import TransformerLM
from underhfs.text import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("underHFS")
    assert tokenizer.decode(tokens) == "underHFS"


def test_transformer_lm_generate_shape():
    model = TransformerLM(vocab_size=256, max_seq_len=8, features=4, hidden_features=8, layers=1)
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("hi")
    generated = model.generate(prompt, max_new_tokens=3)
    assert generated.shape == (5,)
    assert all(0 <= value < 256 for value in generated._storage)
