from datetime import datetime
from typing import List


# ============================================================
# MODULE 1: HOSPITAL STATE MANAGEMENT
# ============================================================

class HospitalState:
    """
    Single source of truth for a hospital's live operational state.
    """

    def __init__(
        self,
        hospital_id: str,
        hospital_name: str,
        total_icu_beds: int,
        total_ward_beds: int,
        opd_daily_capacity: int
    ):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name

        # Bed capacity
        self.total_icu_beds = total_icu_beds
        self.total_ward_beds = total_ward_beds
        self.occupied_icu_beds = 0
        self.occupied_ward_beds = 0

        # OPD capacity
        self.opd_daily_capacity = opd_daily_capacity
        self.opd_tokens_issued = 0

        # Surge status
        self.surge_status = "Normal"  # Normal | High | Critical

        # Timestamp
        self.last_updated = datetime.now()

    # --------------------------------------------------------
    # BED STATE
    # --------------------------------------------------------

    @property
    def icu_beds_available(self) -> int:
        return self.total_icu_beds - self.occupied_icu_beds

    @property
    def ward_beds_available(self) -> int:
        return self.total_ward_beds - self.occupied_ward_beds

    def occupy_icu_bed(self) -> bool:
        if self.icu_beds_available > 0:
            self.occupied_icu_beds += 1
            self._touch()
            return True
        return False

    def occupy_ward_bed(self) -> bool:
        if self.ward_beds_available > 0:
            self.occupied_ward_beds += 1
            self._touch()
            return True
        return False

    # --------------------------------------------------------
    # OPD STATE
    # --------------------------------------------------------

    @property
    def opd_tokens_remaining(self) -> int:
        return self.opd_daily_capacity - self.opd_tokens_issued

    def issue_opd_token(self) -> bool:
        if self.opd_tokens_remaining > 0:
            self.opd_tokens_issued += 1
            self._touch()
            return True
        return False

    # --------------------------------------------------------
    # SURGE CONTROL
    # --------------------------------------------------------

    def set_surge_status(self, status: str):
        if status not in ["Normal", "High", "Critical"]:
            raise ValueError("Invalid surge status")
        self.surge_status = status
        self._touch()

    # --------------------------------------------------------
    # BROADCAST SNAPSHOT
    # --------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Read-only snapshot suitable for public/city-wide broadcast.
        """
        return {
            "hospital_id": self.hospital_id,
            "hospital_name": self.hospital_name,
            "icu_beds_available": self.icu_beds_available,
            "ward_beds_available": self.ward_beds_available,
            "opd_tokens_remaining": self.opd_tokens_remaining,
            "surge_status": self.surge_status,
            "last_updated": self.last_updated.strftime("%Y-%m-%d %H:%M:%S")
        }

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _touch(self):
        self.last_updated = datetime.now()


# ============================================================
# MODULE 0: CITY-WIDE BROADCAST VIEW
# ============================================================

def city_wide_snapshot(hospitals: List[HospitalState]) -> list[dict]:
    """
    Aggregates read-only snapshots from multiple hospitals.
    """
    return [hospital.snapshot() for hospital in hospitals]


# ============================================================
# PHASE 1 DEMO / SANITY CHECK
# ============================================================

if __name__ == "__main__":
    # Create hospitals
    hospital_a = HospitalState("A", "City General Hospital", 5, 20, 60)
    hospital_b = HospitalState("B", "Metro Care Hospital", 2, 15, 50)
    hospital_c = HospitalState("C", "Community Hospital", 1, 10, 40)

    # Simulate some activity
    hospital_a.occupy_icu_bed()
    hospital_a.occupy_ward_bed()

    hospital_b.issue_opd_token()
    hospital_b.issue_opd_token()

    hospital_c.set_surge_status("High")

    # City-wide broadcast
    snapshot = city_wide_snapshot([hospital_a, hospital_b, hospital_c])

    print("\n===== CITY-WIDE HOSPITAL STATUS =====\n")
    for s in snapshot:
        print(s)
