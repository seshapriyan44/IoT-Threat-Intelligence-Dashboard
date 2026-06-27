feature_map = {
    "MI": "Traffic Magnitude",
    "H": "Device Traffic",
    "HH": "Host-to-Host Communication",
    "HpHp": "Port-to-Port Communication",

    "weight": "Traffic Intensity",
    "mean": "Average Traffic Behavior",
    "variance": "Traffic Variability",
    "std": "Traffic Stability",
    "jit": "Packet Timing Jitter"
}


def explain_feature(feature_name):

    explanation = []

    if feature_name.startswith("MI"):
        explanation.append("Traffic Magnitude")

    elif feature_name.startswith("HH"):
        explanation.append("Host-to-Host Communication")

    elif feature_name.startswith("HpHp"):
        explanation.append("Port-to-Port Communication")

    elif feature_name.startswith("H"):
        explanation.append("Device Traffic")

    if "weight" in feature_name:
        explanation.append("Traffic Intensity")

    if "mean" in feature_name:
        explanation.append("Average Traffic Behavior")

    if "variance" in feature_name:
        explanation.append("Traffic Variability")

    if "std" in feature_name:
        explanation.append("Traffic Stability")

    if "jit" in feature_name:
        explanation.append("Packet Timing Jitter")

    # Human-friendly explanations

    if "Traffic Variability" in explanation:
        return (
            "Unusual fluctuations were observed in this traffic pattern "
            "compared to normal device behavior."
        )

    if "Traffic Intensity" in explanation:
        return (
            "Traffic volume was significantly different from the normal baseline."
        )

    if "Port-to-Port Communication" in explanation:
        return (
            "Abnormal communication behavior was detected between network ports."
        )

    if "Host-to-Host Communication" in explanation:
        return (
            "Communication patterns between devices differed from normal activity."
        )

    if "Average Traffic Behavior" in explanation:
        return (
            "Average traffic characteristics deviated from the expected baseline."
        )

    if "Traffic Stability" in explanation:
        return (
            "Traffic stability changed noticeably compared to normal operation."
        )

    if "Packet Timing Jitter" in explanation:
        return (
            "Packet timing became irregular, indicating potential anomalous activity."
        )

    return (
        "This feature showed abnormal behavior compared to the learned baseline."
    )