import json
import time
from pathlib import Path
from typing import Iterable

import requests

from financial_agent.config import get_settings
from financial_agent.schema import FilingMetadata


class SecClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.headers = {
            "User-Agent": self.settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _get_json(self, url: str) -> dict:
        time.sleep(0.2)
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _get_text(self, url: str) -> str:
        time.sleep(0.2)
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.text

    def get_cik_for_ticker(self, ticker: str) -> tuple[str, str]:
        ticker = ticker.strip().upper()

        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get_json(url)

        for item in data.values():
            sec_ticker = str(item.get("ticker", "")).strip().upper()

            if sec_ticker == ticker:
                cik = str(item["cik_str"]).zfill(10)
                company_name = item["title"]
                return cik, company_name

        raise ValueError(
            f"Ticker not found in SEC mapping: {ticker}. "
            "Please check the ticker symbol. Example: Apple is AAPL, not APPL."
        )

    def get_recent_filings(
        self,
        ticker: str,
        forms: Iterable[str] = ("10-K", "10-Q"),
        limit: int = 4,
    ) -> list[FilingMetadata]:
        ticker = ticker.upper()
        wanted_forms = set(forms)

        cik, company_name = self.get_cik_for_ticker(ticker)

        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self._get_json(submissions_url)

        recent = data["filings"]["recent"]

        results: list[FilingMetadata] = []

        for i, form in enumerate(recent["form"]):
            if form not in wanted_forms:
                continue

            accession = recent["accessionNumber"][i]
            primary_document = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]

            accession_no_dashes = accession.replace("-", "")
            cik_no_leading_zeros = str(int(cik))

            url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_leading_zeros}/{accession_no_dashes}/{primary_document}"
            )

            results.append(
                FilingMetadata(
                    ticker=ticker,
                    cik=cik,
                    company_name=company_name,
                    form=form,
                    filing_date=filing_date,
                    accession_number=accession,
                    primary_document=primary_document,
                    url=url,
                )
            )

            if len(results) >= limit:
                break

        return results

    def download_filing(
        self,
        filing: FilingMetadata,
        output_dir: str | Path = "data/raw",
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = (
            f"{filing.ticker}_"
            f"{filing.form}_"
            f"{filing.filing_date}_"
            f"{filing.accession_number}.html"
        ).replace("/", "-")

        output_path = output_dir / safe_name

        if output_path.exists():
            return output_path

        html = self._get_text(filing.url)
        output_path.write_text(html, encoding="utf-8")

        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(filing.__dict__, indent=2),
            encoding="utf-8",
        )

        return output_path
