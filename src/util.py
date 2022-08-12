
def autoconfigure():
    import os, sys
    from pathlib import Path
    new = str(Path(os.path.realpath(__file__)).parent.parent)
    if new not in sys.path:
        sys.path.append(new)

    for obj in os.listdir(new):
        path = Path(new) / obj
        full_path = str(path)
        if not path.is_dir() \
        or obj.startswith('.') \
        or obj.startswith('__'):
            continue
        print(full_path)
        if full_path not in sys.path:
            sys.path.append(full_path)

    print(sys.path)


def df_to_list(df):
    cols = df.columns
    X = []
    for row in df.itertuples(index=False):
        X.append([row[col] for col in cols])
    return X