def group_by_perturbation_type(records: list) -> dict:
    grouped = {}
    for record in records:
        perturbation_type = record.get("perturbation_type", None)
        if perturbation_type not in grouped:
            grouped[perturbation_type] = []
        grouped[perturbation_type].append(record)
    return grouped
