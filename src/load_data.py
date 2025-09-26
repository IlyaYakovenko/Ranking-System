import pandas as pd

def load_letor_data(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')[0].split()
            label = int(parts[0])
            qid = int(parts[1].split(':')[1])
            features = {f'feature_{i}': 0.0 for i in range(1, 47)}

            for elem in parts[2:]:
                if ':' in elem:
                    fid, value = elem.split(':')
                    fid = int(fid)
                    if 1 <= fid <= 46:
                        features[f'feature_{fid}'] = float(value)
                    else:
                        print(f"Недопустимый признак: elem={elem}, fid={fid}")

            row = {'label': label, 'qid': qid}
            row.update(features)
            data.append(row)

    df = pd.DataFrame(data)
    return df