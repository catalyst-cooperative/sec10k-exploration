import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SEC-10k exploration
    This notebook demonstrates how to access SEC-10k data and the attached exhibit 21 filings, which contain information about the subsidiary companies that a filing company owns. It also does some basic prototyping to demonstrate the use of an LLM for extracting ownership information from exhibit 21 filings. The preliminary results for this extraction are very encouraging, but more validation and testing is required.

    To perform the LLM extraction yourself, you just need an [openrouter](https://openrouter.ai/) account and API key. Once you have acquired one, you should be able to set an environment variable called `OPENROUTER_API_KEY` and this entire notebook should run without issue. The one caveat is that the SEC website may rate limit you if you try to download too many filings at once, so be careful if you set the range of filings to test with to be too high.
    """)
    return


@app.cell
def _():
    import gzip
    from io import BytesIO

    import pandas as pd
    import requests

    def download_sec_master_files(year_quarters):
        """Download and concatenate SEC master index files.

        Parameters
        ----------
        year_quarters : iterable of tuple[int, int]
            Year/quarter pairs, such as {(2023, 1), (2023, 2)}.

        Returns
        -------
        pandas.DataFrame
            The concatenated master index records, with a ``year`` and ``quarter``
            column identifying the source file.
        """
        columns = ["cik", "company_name", "form_type", "date_filed", "filename"]
        dataframes = []
        session = requests.Session()
        session.headers.update({"User-Agent": "research contact@example.com"})

        for year, quarter in sorted(set(year_quarters)):
            year = int(year)
            quarter = int(quarter)
            if quarter not in {1, 2, 3, 4}:
                raise ValueError(f"Quarter must be 1, 2, 3, or 4; received {quarter!r}")

            url = (
                f"https://www.sec.gov/Archives/edgar/full-index/"
                f"{year}/QTR{quarter}/master.gz"
            )
            response = session.get(url, timeout=60)
            response.raise_for_status()

            with gzip.GzipFile(fileobj=BytesIO(response.content)) as master_file:
                master_df = pd.read_csv(
                    master_file,
                    sep="|",
                    skiprows=11,
                    names=columns,
                    dtype="string",
                    keep_default_na=False,
                )

            master_df["year"] = year
            master_df["quarter"] = quarter
            dataframes.append(master_df)

        if not dataframes:
            return pd.DataFrame(columns=columns + ["year", "quarter"])

        return pd.concat(dataframes, ignore_index=True)

    return download_sec_master_files, pd, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downloading filings
    Below we will define a function to download exhibit 21 filings. For demonstration purposes, we download a couple quarters of filings and grab a selection for display. If you click on the `html` in the `ex21_text` column, you can see a rendered version of the exhibit 21. Notice how inconsistent the formatting is.
    """)
    return


@app.cell
def _(download_sec_master_files, requests):
    import re

    def get_sampled_ex21_filings(year_quarters, sample_size=10, random_state=42):
        """Download sampled 10-K filings and extract EX-21.1 document blocks.

        year_quarters : iterable of tuple[int, int]
            Year/quarter pairs, such as {(2023, 1), (2023, 2)}.
        """
        master_records = download_sec_master_files(year_quarters)
        ten_k_records = master_records.loc[
            master_records["form_type"].eq("10-K")
        ].copy()

        sample_count = min(sample_size, len(ten_k_records))
        sampled_records = ten_k_records.sample(
            n=sample_count,
            random_state=random_state,
        ).copy()

        session = requests.Session()
        session.headers.update({"User-Agent": "research contact@example.com"})

        def extract_ex21_text(filename):
            url = f"https://www.sec.gov/Archives/{filename}"
            response = session.get(url, timeout=60)
            response.raise_for_status()
            match = re.search(
                r"<DOCUMENT>\s*<TYPE>EX-21\.1.*?</DOCUMENT>",
                response.text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return match.group(0) if match else None

        sampled_records["ex21_text"] = sampled_records["filename"].map(
            extract_ex21_text
        )
        return sampled_records.loc[sampled_records["ex21_text"].notna()].copy()

    get_sampled_ex21_filings({(2023, 1), (2023, 2)})
    return (get_sampled_ex21_filings,)


@app.cell
def _():
    from pydantic import BaseModel, ConfigDict

    class Subsidiary(BaseModel):
        """Define schema for a subsidiary in exhibit 21."""

        model_config = ConfigDict(extra="forbid")

        company_name: str
        location_of_incorporation: str | None
        fraction_owned_by_parent: float | None
        immediate_parent: str | None

    class SubsidiaryList(BaseModel):
        """List of subsidiary companies found in exhibit 21."""

        model_config = ConfigDict(extra="forbid")

        subsidiary_list: list[Subsidiary]

    return (SubsidiaryList,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OpenRouter extraction setup

    This cell configures an OpenRouter client and prepares instructions for the model to extract exhibit 21 filings. There is a markdown file in this repo called `extract_instructions.md` and a JSON schema defined in the code for the return structure. This is all that is needed to produce consistent and accurate results from the extraction (at least in preliminary testing).
    """)
    return


@app.cell
def _(SubsidiaryList):
    import json
    import os
    from pathlib import Path

    from openai import OpenAI

    instructions_path = Path("extract_instructions.md")
    extraction_instructions = instructions_path.read_text(encoding="utf-8")

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise RuntimeError(
            "Set the OPENROUTER_API_KEY environment variable before running extraction."
        )

    openrouter_client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.environ.get(
                "OPENROUTER_APP_NAME", "SEC subsidiary extractor"
            ),
        },
    )

    openrouter_model = os.environ.get(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini",
    )

    def validate_subsidiary_list(response_json):
        if hasattr(SubsidiaryList, "model_validate"):
            return SubsidiaryList.model_validate(response_json)
        return SubsidiaryList.parse_obj(response_json)

    def extract_subsidiaries_from_ex21(ex_21_text, *, temperature=0):
        """Extract and validate subsidiaries from one EX-21 document block."""
        if not ex_21_text or not str(ex_21_text).strip():
            return None

        completion = openrouter_client.chat.completions.create(
            model=openrouter_model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract subsidiary information from SEC Exhibit 21 filings.\n\n"
                        f"{extraction_instructions}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the subsidiaries from the following EX-21 document block. "
                        "Return only data matching the supplied JSON schema.\n\n"
                        f"{ex_21_text}"
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "subsidiary_list",
                    "strict": True,
                    "schema": SubsidiaryList.model_json_schema(),
                },
            },
        )

        content = completion.choices[0].message.content
        if not content:
            raise ValueError("OpenRouter returned an empty response.")

        parsed_result = validate_subsidiary_list(json.loads(content))

        usage = completion.usage
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)

        return parsed_result, usage_dict

    return (extract_subsidiaries_from_ex21,)


@app.cell
def _(mo):
    from datetime import date

    # Select the inclusive start and stop years and the number of filings to sample.
    start_year_input = mo.ui.number(
        start=1994,
        stop=date.today().year,
        value=2023,
        step=1,
        label="Start year",
    )
    stop_year_input = mo.ui.number(
        start=1994,
        stop=date.today().year,
        value=2023,
        step=1,
        label="Stop year",
    )
    sample_size_input = mo.ui.number(
        start=1,
        stop=1000,
        value=20,
        step=1,
        label="Number of filings to sample from date range",
    )
    return date, sample_size_input, start_year_input, stop_year_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select Example Data
    Below select a range of years to grab filings from plus a sample size. This will download all ex21 filings for the selected years and grab a sample to actually send to the LLM. Keeping the year range and sample size relatively limited will help to avoid excessive runtime.
    """)
    return


@app.cell
def _(
    date,
    extract_subsidiaries_from_ex21,
    get_sampled_ex21_filings,
    mo,
    sample_size_input,
    start_year_input,
    stop_year_input,
):
    current_year = date.today().year
    current_quarter = (date.today().month - 1) // 3 + 1
    year_quarters = {
        (year, quarter)
        for year in range(int(start_year_input.value), int(stop_year_input.value) + 1)
        for quarter in range(
            1,
            (current_quarter if year == current_year else 4) + 1,
        )
    }

    # Get filing samples for every quarter in the selected year range.
    sampled_ex21_filings = get_sampled_ex21_filings(
        year_quarters,
        sample_size=int(sample_size_input.value),
    )

    # Run the extraction over the sampled filings.
    # This makes one OpenRouter request per non-empty EX-21 document block.
    for column_name in [
        "parsed_result",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error",
    ]:
        sampled_ex21_filings[column_name] = None

    for row_index, ex_21_text in sampled_ex21_filings["ex21_text"].items():
        try:
            parsed_result, usage = extract_subsidiaries_from_ex21(ex_21_text)

            sampled_ex21_filings.loc[
                row_index,
                [
                    "parsed_result",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "error",
                ],
            ] = [
                parsed_result,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                None,
            ]
        except Exception as error:
            sampled_ex21_filings.loc[
                row_index,
                [
                    "parsed_result",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "error",
                ],
            ] = [None, None, None, None, repr(error)]

    mo.vstack(
        [
            mo.hstack(
                [start_year_input, stop_year_input, sample_size_input], widths="equal"
            ),
            mo.ui.table(sampled_ex21_filings),
        ]
    )
    return (sampled_ex21_filings,)


@app.cell
def _(mo, sampled_ex21_filings):
    company_options = (
        sampled_ex21_filings["company_name"].dropna().drop_duplicates().tolist()
    )
    company_selector = mo.ui.dropdown(
        options=company_options,
        value=company_options[0] if company_options else None,
        label="Select a company",
    )
    return (company_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Extraction
    Next we will actually send data to the model for extraction. Once this is complete, you will be able to select a company name from the sample of filings and see rendered exhibit 21 `html` next to the extracted subsidiaries. This allows us to manually spot check the results.
    """)
    return


@app.cell
def _(company_selector, mo, pd, sampled_ex21_filings):
    from html import escape

    selected_company = company_selector.value
    selected_rows = sampled_ex21_filings.loc[
        sampled_ex21_filings["company_name"].eq(selected_company)
    ]

    if selected_rows.empty:
        selected_output = mo.md("No filing is available for the selected company.")
    else:
        selected_filing = selected_rows.iloc[0]
        ex21_html = str(selected_filing["ex21_text"] or "")

        rendered_ex21 = mo.Html(
            f'<iframe srcdoc="{escape(ex21_html, quote=True)}" '
            'style="width:100%; height:650px; border:1px solid #d1d5db; '
            'border-radius:6px; background:white;"></iframe>'
        )

        parsed_subsidiaries = selected_filing["parsed_result"]
        subsidiary_records = []

        if parsed_subsidiaries is not None:
            subsidiary_items = (
                parsed_subsidiaries.get("subsidiary_list", [])
                if isinstance(parsed_subsidiaries, dict)
                else getattr(parsed_subsidiaries, "subsidiary_list", [])
            )

            for subsidiary in subsidiary_items:
                if isinstance(subsidiary, dict):
                    subsidiary_records.append(subsidiary)
                else:
                    subsidiary_records.append(
                        {
                            "company_name": subsidiary.company_name,
                            "location_of_incorporation": (
                                subsidiary.location_of_incorporation
                            ),
                            "fraction_owned_by_parent": (
                                subsidiary.fraction_owned_by_parent
                            ),
                            "immediate_parent": subsidiary.immediate_parent,
                        }
                    )

        subsidiary_table = pd.DataFrame(
            subsidiary_records,
            columns=[
                "company_name",
                "location_of_incorporation",
                "fraction_owned_by_parent",
                "immediate_parent",
            ],
        )

        rendered_subsidiaries = mo.ui.table(subsidiary_table)

        selected_output = mo.vstack(
            [
                mo.md("## Overview"),
                mo.md(
                    f"**Average Input Tokens:** {sampled_ex21_filings['prompt_tokens'].mean():.2f}, "
                    f"**Average Output Tokens:** {sampled_ex21_filings['completion_tokens'].mean():.2f}, "
                ),
                mo.md("## Inspect Filings"),
                company_selector,
                mo.md(f"### {selected_company}"),
                mo.md(
                    f"**Tokens:** {selected_filing['prompt_tokens']} input, "
                    f"{selected_filing['completion_tokens']} output"
                ),
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.md("#### Rendered EX-21 HTML"),
                                rendered_ex21,
                            ]
                        ),
                        mo.vstack(
                            [
                                mo.md("#### Extracted subsidiaries"),
                                rendered_subsidiaries,
                            ]
                        ),
                    ],
                    widths="equal",
                    gap=1,
                ),
            ]
        )

    selected_output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
