"""One complete solution. The independent task uses different data/order."""


def compare_pairs(baseline, candidate):
    def index(rows):
        result = {}
        for row in rows:
            if (not isinstance(row.get("case_id"), str) or not row["case_id"]
                    or row["case_id"] in result or type(row.get("score")) is not int
                    or not 0 <= row["score"] <= 2 or type(row.get("critical")) is not bool):
                raise ValueError("invalid observation")
            result[row["case_id"]] = row
        if not result:
            raise ValueError("empty observation set")
        return result
    a, b = index(baseline), index(candidate)
    if set(a) != set(b):
        raise ValueError("unpaired observations")
    return {"improved": sorted(k for k in a if b[k]["score"] > a[k]["score"]),
            "regressed": sorted(k for k in a if b[k]["score"] < a[k]["score"]),
            "critical_regressions": sorted(k for k in a if b[k]["critical"] and not a[k]["critical"]),
            "mean_delta": sum(b[k]["score"] - a[k]["score"] for k in a) / len(a)}
