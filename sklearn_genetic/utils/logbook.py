def logbook_to_pandas(logbook, chapter="parameters", fields=None, include_index=True):
    import pandas as pd

    data = logbook.chapters[chapter]
    df = pd.DataFrame(data)

    if not include_index and "index" in df.columns:
        df = df.drop(columns=["index"])

    if fields is not None:
        available_fields = [field for field in fields if field in df.columns]
        df = df[available_fields]

    return df
