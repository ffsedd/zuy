from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Sample:
    """Represents a sample within a Zakazka."""

    zakazka: int
    sample_no: int


@dataclass(frozen=True, order=True)
class MeasurementSpot:
    """Represents a measurement spot on a specific sample."""

    sample: Sample
    spot_no: int


if __name__ == "__main__":
    # Example usage
    s = Sample(zakazka=1, sample_no=2)
    m = MeasurementSpot(sample=s, spot_no=3)

    print(s)  # Sample(zakazka=1, sample_no=2)
    print(m)  # MeasurementSpot(sample=Sample(zakazka=1, sample_no=2), spot_no=3)
