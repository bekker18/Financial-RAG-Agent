from financial_agent.chunking import chunk_text, clean_text


def test_clean_text_removes_extra_whitespace():
    raw_text = "Apple   reports\n\nstrong\trevenue.   Risk\xa0factors exist."
    cleaned = clean_text(raw_text)

    assert cleaned == "Apple reports strong revenue. Risk factors exist."


def test_chunk_text_returns_chunks():
    text = "Apple business risk. " * 300

    metadata = {
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2024-11-01",
        "accession_number": "0000320193-24-000123",
    }

    chunks = chunk_text(
        text=text,
        metadata=metadata,
        chunk_size=500,
        overlap=100,
    )

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.id for chunk in chunks)


def test_chunk_text_preserves_metadata():
    text = "Microsoft cloud revenue risk. " * 100

    metadata = {
        "ticker": "MSFT",
        "form": "10-Q",
        "filing_date": "2024-10-30",
        "accession_number": "0000789019-24-000456",
    }

    chunks = chunk_text(
        text=text,
        metadata=metadata,
        chunk_size=300,
        overlap=50,
    )

    first_chunk = chunks[0]

    assert first_chunk.metadata["ticker"] == "MSFT"
    assert first_chunk.metadata["form"] == "10-Q"
    assert first_chunk.metadata["filing_date"] == "2024-10-30"
    assert first_chunk.metadata["accession_number"] == "0000789019-24-000456"
    assert first_chunk.metadata["chunk_index"] == 0


def test_chunk_text_empty_input_returns_empty_list():
    chunks = chunk_text(
        text="   \n\t   ",
        metadata={"ticker": "AAPL"},
        chunk_size=500,
        overlap=100,
    )

    assert chunks == []


def test_chunk_text_overlap_works():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20

    chunks = chunk_text(
        text=text,
        metadata={
            "ticker": "TEST",
            "form": "10-K",
            "filing_date": "2024-01-01",
            "accession_number": "test-accession",
        },
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    first_chunk_end = chunks[0].text[-20:]
    second_chunk_start_area = chunks[1].text[:30]

    assert first_chunk_end in second_chunk_start_area
