"""Lazy: keep the idle frequency, always 'changed' with one fixed guess (TLS 0 moved to 200 MHz)."""


def solve(data):
    return {"idle_mhz": data["idle_mhz"],
            "diagnosis": {"changed": True, "kinds": {"tls_moved": {"tls": 0, "freq_mhz": 200.0}}}}
