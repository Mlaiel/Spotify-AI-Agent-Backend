"""
Module: helpers.py
Description: Fonctions utilitaires transverses (flatten, chunk, merge, deep_get, safe_cast, etc).
"""
def flatten(lst):
    return [item for sublist in lst for item in sublist]

def chunk(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def deep_get(dct, keys, default=None):
    for key in keys:
        if isinstance(dct, dict):
            dct = dct.get(key, default)
        else:
            return default
    return dct

def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except Exception:
        return default

# Exemples d'utilisation
# flatten([1,2],[3,4])
# chunk([1,2,3,4,5], 2)
# deep_get({"a": {"b": 2}, ["a", "b"])
# safe_cast("42", int)
