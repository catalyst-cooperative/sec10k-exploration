# SEC-10k / Exhibit 21 Explorer
This repo contains a [marimo](https://docs.marimo.io/#quickstart) notebook for exploring SEC-10k
data and the attached Exhibit 21 documents. It demonstrates how to access the filings, and prototypes
extracting company ownership from the Exhibit 21s using an LLM through `openrouter`.

## Usage
Running the notebook should be fairly straightforward. You need to install `uv` and get an `openrouter`
API key.

### Steps
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Signup for [openrouter]. (the notebook uses the `openai` API, so this could be adapted to work with 
any model provider that provides a compatible API)
3. Set `OPENROUTER_API_KEY` environment variable
4. Run `uv run marimo run sec10k_explorer.py`
5. Wait several minutes for initial run to complete
