import re
import ast

raw = """\"{'pt1': (np.float64(42.393167), np.float64(-71.064352)), 'pt2': (np.float64(40.718037), np.float64(-73.932309)), 'dist': 340157, 'time': 16639, 'bat_H': np.float64(1573035721.0281672), 'bat_L': np.float64(938686773.501855)}\",\"{'pt1': (np.float64(42.393167), np.float64(-71.064352)), 'pt2': (np.float64(41.527246), np.float64(-72.064419)), 'dist': 159537, 'time': 7641, 'bat_H': np.float64(748493820.7796407), 'bat_L': np.float64(449470255.0766783)}\""""

def parse_np_dict_string(s: str) -> list[dict]:
    # Split on the boundary between quoted dict-strings: ","
    parts = s.split('","')
    result = []
    for part in parts:
        part = part.strip().strip('"')
        if not part:
            continue
        # Replace np.float64(x) → x so ast.literal_eval can handle it
        part = re.sub(r'np\.float64\(([^)]+)\)', r'\1', part)
        try:
            result.append(ast.literal_eval(part))
        except (ValueError, SyntaxError):
            pass
    return result


if __name__ == "__main__":
    records = parse_np_dict_string(raw)
    print(f"Parsed {len(records)} records")
    for r in records:
        print(r)
