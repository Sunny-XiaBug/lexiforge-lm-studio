import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import streamlit as st
from nltk.util import ngrams as nltk_ngrams


APP_NAME = "LexiForge LM Studio"
DEFAULT_CORPUS = """The market opened with cautious optimism as investors watched the technology sector.
Analysts said the new language model platform could improve customer support workflows.
The company reported stronger demand in cloud services and artificial intelligence tooling.
Engineers tested the model with clean evaluation data before releasing the dashboard."""

DEFAULT_RNN_CORPUS = "hello world hello world hello streamlit hello world "


st.set_page_config(page_title=APP_NAME, layout="wide")


def word_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[.,!?;:]", text.lower())


@dataclass
class NGramModel:
    n: int
    vocabulary: List[str]
    ngram_counts: Counter
    context_counts: Counter


def build_ngram_model(corpus: str, n: int = 3) -> NGramModel:
    tokens = ["<s>"] * (n - 1) + word_tokenize(corpus) + ["</s>"]
    ngram_counts = Counter(nltk_ngrams(tokens, n))
    context_counts = Counter(tuple(tokens[i : i + n - 1]) for i in range(len(tokens) - n + 1))
    vocabulary = sorted(set(tokens))
    return NGramModel(n=n, vocabulary=vocabulary, ngram_counts=ngram_counts, context_counts=context_counts)


def score_sentence(model: NGramModel, sentence: str, smooth: bool) -> Tuple[float, float, pd.DataFrame]:
    tokens = ["<s>"] * (model.n - 1) + word_tokenize(sentence) + ["</s>"]
    rows = []
    log_prob = 0.0
    probability = 1.0
    vocab_size = len(model.vocabulary)

    for i in range(len(tokens) - model.n + 1):
        ngram = tuple(tokens[i : i + model.n])
        context = ngram[:-1]
        count = model.ngram_counts[ngram]
        context_count = model.context_counts[context]

        if smooth:
            numerator = count + 1
            denominator = context_count + vocab_size
        else:
            numerator = count
            denominator = context_count

        step_prob = numerator / denominator if denominator else 0.0
        probability *= step_prob
        log_prob += math.log(step_prob) if step_prob > 0 else float("-inf")
        rows.append(
            {
                "n-gram": " ".join(ngram),
                "context": " ".join(context),
                "count": count,
                "context_count": context_count,
                "probability": step_prob,
                "status": "seen" if count else "unseen",
            }
        )

    return probability, log_prob, pd.DataFrame(rows)


def format_probability(value: float) -> str:
    if value == 0 or not math.isfinite(value):
        return "0"
    return f"{value:.3e}"


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"PyTorch is not available: {exc}")
        st.stop()


def train_char_rnn(corpus: str, hidden_size: int, epochs: int, learning_rate: float):
    torch, nn, _ = import_torch()

    class CharRNN(nn.Module):
        def __init__(self, vocab_size: int, hidden_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            embedded = self.embedding(x)
            output, _ = self.rnn(embedded)
            return self.head(output)

    chars = sorted(set(corpus))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    encoded = torch.tensor([char_to_idx[c] for c in corpus], dtype=torch.long)
    x = encoded[:-1].unsqueeze(0)
    y = encoded[1:].unsqueeze(0)

    torch.manual_seed(7)
    model = CharRNN(len(chars), hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    chart = st.line_chart(pd.DataFrame({"loss": []}))
    progress = st.progress(0, text="Training character-level language model...")
    losses = []
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        chart.add_rows(pd.DataFrame({"loss": [loss_value]}, index=[epoch]))
        progress.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs} | loss {loss_value:.4f}")

    progress.empty()
    return model, char_to_idx, idx_to_char, losses


def generate_with_rnn(model, seed: str, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str], length: int, temperature: float):
    torch, _, F = import_torch()
    model.eval()
    current = seed[-1] if seed else next(iter(char_to_idx))
    if current not in char_to_idx:
        current = next(iter(char_to_idx))

    output = seed or current
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([[char_to_idx[current]]], dtype=torch.long)
            embedded = model.embedding(x)
            result, hidden = model.rnn(embedded, hidden)
            logits = model.head(result[:, -1, :]) / max(temperature, 0.05)
            probs = F.softmax(logits, dim=-1)
            next_idx = int(torch.multinomial(probs, num_samples=1).item())
            current = idx_to_char[next_idx]
            output += current
    return output


@st.cache_resource(show_spinner=False)
def load_fill_mask_pipeline():
    from transformers import pipeline

    return pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_gpt2_generation_pipeline():
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    return generator


@st.cache_resource(show_spinner=False)
def load_gpt2_for_ppl():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model, torch


def compute_ppl(sentences: Sequence[str]) -> pd.DataFrame:
    tokenizer, model, torch = load_gpt2_for_ppl()
    rows = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        if inputs["input_ids"].shape[1] < 2:
            rows.append({"sentence": sentence, "cross_entropy": None, "perplexity": None})
            continue
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = float(outputs.loss.item())
        rows.append({"sentence": sentence, "cross_entropy": loss, "perplexity": math.exp(loss)})
    return pd.DataFrame(rows)


st.title(APP_NAME)
st.caption("Interactive language-model training, generation, architecture comparison, and perplexity analysis.")

tab_ngram, tab_rnn, tab_pretrained, tab_ppl = st.tabs(
    ["N-gram & Smoothing", "Train RNN-LM", "BERT vs GPT-2", "Perplexity"]
)

with tab_ngram:
    st.subheader("N-gram Language Model with Laplace Smoothing")
    left, right = st.columns([1.05, 0.95])
    with left:
        corpus = st.text_area("Training corpus", DEFAULT_CORPUS, height=210)
        n_value = st.selectbox("Model order", [2, 3], index=1, format_func=lambda x: "Bigram" if x == 2 else "Trigram")
        sentence = st.text_input("Sentence to score", "the company released a new model")
        use_smoothing = st.checkbox("Use add-one / Laplace smoothing", value=True)
    with right:
        model = build_ngram_model(corpus, n_value)
        raw_prob, raw_log, raw_table = score_sentence(model, sentence, smooth=False)
        smooth_prob, smooth_log, smooth_table = score_sentence(model, sentence, smooth=True)
        active_prob = smooth_prob if use_smoothing else raw_prob
        active_log = smooth_log if use_smoothing else raw_log

        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulary", len(model.vocabulary))
        c2.metric("Unique n-grams", len(model.ngram_counts))
        c3.metric("Active probability", format_probability(active_prob))
        st.metric("Log probability", "−∞" if active_log == float("-inf") else f"{active_log:.4f}")

        comparison = pd.DataFrame(
            [
                {
                    "mode": "No smoothing",
                    "joint_probability": format_probability(raw_prob),
                    "log_probability": "−∞" if raw_log == float("-inf") else f"{raw_log:.4f}",
                },
                {
                    "mode": "Add-one smoothing",
                    "joint_probability": format_probability(smooth_prob),
                    "log_probability": "−∞" if smooth_log == float("-inf") else f"{smooth_log:.4f}",
                },
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.dataframe((smooth_table if use_smoothing else raw_table), use_container_width=True, hide_index=True)

with tab_rnn:
    st.subheader("Train a Character-level RNN Language Model")
    train_col, result_col = st.columns([0.95, 1.05])
    with train_col:
        rnn_corpus = st.text_area("Custom training corpus", DEFAULT_RNN_CORPUS, height=180)
        hidden_size = st.slider("Hidden size", 16, 128, 64, step=16)
        epochs = st.slider("Epochs", 10, 200, 80, step=10)
        learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.02, step=0.001, format="%.3f")
        train = st.button("Start training", type="primary")

    with result_col:
        if train:
            if len(set(rnn_corpus)) < 2 or len(rnn_corpus) < 8:
                st.warning("Please provide at least 8 characters and 2 unique characters.")
            else:
                model, char_to_idx, idx_to_char, losses = train_char_rnn(rnn_corpus, hidden_size, epochs, learning_rate)
                st.session_state["rnn_artifacts"] = {
                    "model": model,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "losses": losses,
                }
                st.success(f"Training complete. Final loss: {losses[-1]:.4f}")

        artifacts = st.session_state.get("rnn_artifacts")
        if artifacts:
            seed = st.text_input("Seed text", "h")
            length = st.slider("Generated length", 20, 160, 60, step=10)
            temperature = st.slider("Sampling temperature", 0.2, 1.5, 0.8, step=0.1)
            if st.button("Generate text"):
                generated = generate_with_rnn(
                    artifacts["model"],
                    seed,
                    artifacts["char_to_idx"],
                    artifacts["idx_to_char"],
                    length,
                    temperature,
                )
                st.code(generated)
            st.line_chart(pd.DataFrame({"loss": artifacts["losses"]}))
        else:
            st.info("Train a model to unlock text generation.")

with tab_pretrained:
    st.subheader("Pretrained Architecture Comparison")
    bert_col, gpt_col = st.columns(2)
    with bert_col:
        st.markdown("**BERT masked language modeling**")
        masked_sentence = st.text_input(
            "Input with [MASK]",
            "The man went to the [MASK] to buy some milk.",
        )
        if st.button("Predict mask"):
            if "[MASK]" not in masked_sentence:
                st.warning("BERT expects one [MASK] token in the sentence.")
            else:
                with st.spinner("Loading BERT and predicting candidates..."):
                    fill_mask = load_fill_mask_pipeline()
                    predictions = fill_mask(masked_sentence, top_k=5)
                st.dataframe(
                    pd.DataFrame(
                        {
                            "token": [p["token_str"].strip() for p in predictions],
                            "score": [p["score"] for p in predictions],
                            "sequence": [p["sequence"] for p in predictions],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with gpt_col:
        st.markdown("**GPT-2 causal language modeling**")
        prompt = st.text_area("Prompt prefix", "Language models can help engineers", height=110)
        max_new_tokens = st.slider("New tokens", 10, 60, 25, step=5)
        if st.button("Generate continuation"):
            with st.spinner("Loading GPT-2 and generating continuation..."):
                generator = load_gpt2_generation_pipeline()
                result = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    num_return_sequences=1,
                    pad_token_id=generator.tokenizer.eos_token_id,
                )[0]["generated_text"]
            st.write(result)

with tab_ppl:
    st.subheader("GPT-2 Perplexity Scoring")
    sample_text = """The engineering team released a reliable language model dashboard.
apple quantum river loudly dashboard glass running"""
    ppl_input = st.text_area("Sentences, one per line", sample_text, height=170)
    if st.button("Compute perplexity", type="primary"):
        sentences = [line.strip() for line in ppl_input.splitlines() if line.strip()]
        if not sentences:
            st.warning("Please enter at least one sentence.")
        else:
            with st.spinner("Loading GPT-2 and calculating cross-entropy loss..."):
                results = compute_ppl(sentences)
            st.dataframe(results, use_container_width=True, hide_index=True)
