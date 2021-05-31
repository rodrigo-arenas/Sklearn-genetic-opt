import pandas as pd


def logbook_to_pandas(logbook):
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df
