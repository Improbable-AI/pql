from pathlib import Path

from pql.utils.common import list_class_names

cur_path = Path(__file__).resolve().parent
alg_name_to_path = list_class_names(cur_path)
