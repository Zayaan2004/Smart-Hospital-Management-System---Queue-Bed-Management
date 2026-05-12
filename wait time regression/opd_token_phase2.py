from datetime import datetime, timedelta
import uuid


# ============================================================
# PHASE 2 CONFIGURATION
# ============================================================

AVG_SERVICE_TIME_MIN = 13          # minutes per patient
SURGE_MULTIPLIER = 1.5             # allowed burst
TOKEN_EXPIRY_MIN = 30              # minutes to scan QR

# OPD HOURS
OPD_START_HOUR = 10
OPD_END_HOUR = 19

# Nearby hospitals for redirection
NEARBY_HOSPITALS = [
    "Hospital B (Low Load)",
    "Hospital C (Moderate Load)"
]


# ============================================================
# DERIVED CAPACITY (PER HOUR)
# ============================================================

SERVICE_RATE_PER_HOUR = int(60 / AVG_SERVICE_TIME_MIN)
MAX_TOKENS_PER_HOUR = int(SERVICE_RATE_PER_HOUR * SURGE_MULTIPLIER)


# ============================================================
# RUNTIME STATE (IN-MEMORY)
# ============================================================

issued_tokens = []        # all tokens (pending + active + served)
active_queue = []         # scanned tokens only
hourly_issue_log = []     # timestamps of issued tokens


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def opd_is_open():
    hour = datetime.now().hour
    return OPD_START_HOUR <= hour < OPD_END_HOUR


def clean_hourly_log():
    """Keep only last 60 minutes of issuance data"""
    cutoff = datetime.now() - timedelta(hours=1)
    while hourly_issue_log and hourly_issue_log[0] < cutoff:
        hourly_issue_log.pop(0)


def surge_level():
    clean_hourly_log()
    issued_last_hour = len(hourly_issue_log)

    if issued_last_hour > MAX_TOKENS_PER_HOUR * 1.2:
        return "CRITICAL"
    elif issued_last_hour > MAX_TOKENS_PER_HOUR:
        return "HIGH"
    else:
        return "NORMAL"


def estimated_wait_time():
    base_wait = len(active_queue) * AVG_SERVICE_TIME_MIN
    level = surge_level()

    if level == "HIGH":
        return int(base_wait * 1.5)
    if level == "CRITICAL":
        return int(base_wait * 2.5)

    return base_wait


# ============================================================
# TOKEN GENERATION (SELF-SERVICE)
# ============================================================

def generate_token():
    clean_hourly_log()

    if len(hourly_issue_log) >= MAX_TOKENS_PER_HOUR:
        return None, "RATE_LIMIT"

    token_id = str(uuid.uuid4())[:8]
    token_number = len(issued_tokens) + 1

    token = {
        "id": token_id,
        "number": token_number,
        "status": "PENDING",
        "issued_at": datetime.now(),
        "activated_at": None
    }

    issued_tokens.append(token)
    hourly_issue_log.append(datetime.now())

    return token, None


# ============================================================
# TOKEN ACTIVATION (ENTRANCE SCAN)
# ============================================================

def activate_token(token_id):
    for token in issued_tokens:
        if token["id"] == token_id:
            if token["status"] != "PENDING":
                return False, "INVALID_STATE"

            if datetime.now() - token["issued_at"] > timedelta(minutes=TOKEN_EXPIRY_MIN):
                token["status"] = "EXPIRED"
                return False, "EXPIRED"

            token["status"] = "ACTIVE"
            token["activated_at"] = datetime.now()
            active_queue.append(token)
            return True, None

    return False, "NOT_FOUND"


# ============================================================
# SERVING PATIENTS
# ============================================================

def serve_next_patient():
    if not active_queue:
        return None

    token = active_queue.pop(0)
    token["status"] = "SERVED"
    return token["number"]


# ============================================================
# USER INTERFACE (SIMULATION)
# ============================================================

print("\n===== OPD SELF-SERVICE TOKEN SYSTEM =====\n")
print(f"Avg Service Time        : {AVG_SERVICE_TIME_MIN} minutes")
print(f"Service Rate            : {SERVICE_RATE_PER_HOUR} patients/hour")
print(f"Token Burst Limit       : {MAX_TOKENS_PER_HOUR} tokens/hour")
print("Tokens become valid only after QR scan at entrance\n")

while opd_is_open():
    print("\n--- OPD STATUS ---")
    print(f"Active Queue Length     : {len(active_queue)}")
    print(f"Estimated Wait Time     : {estimated_wait_time()} minutes")
    print(f"Surge Level             : {surge_level()}")

    if surge_level() in ["HIGH", "CRITICAL"]:
        print("\n⚠ HIGH OPD LOAD")
        print("Consider nearby hospitals:")
        for h in NEARBY_HOSPITALS:
            print(f"• {h}")

    print("\nOptions:")
    print("1. Generate OPD token (patient)")
    print("2. Scan QR at entrance (activate token)")
    print("3. Serve next patient")
    print("4. Exit system")

    choice = input("Choose option (1–4): ").strip()

    if choice == "1":
        token, error = generate_token()
        if token:
            print("\n✅ TOKEN GENERATED")
            print(f"Token Number : {token['number']}")
            print(f"QR Code ID   : {token['id']}")
            print(f"Est. Wait    : {estimated_wait_time()} minutes")
        else:
            print("\n❌ TOKEN GENERATION PAUSED")
            print("OPD is under heavy load. Please try again later.")

    elif choice == "2":
        tid = input("Enter QR Token ID: ").strip()
        success, reason = activate_token(tid)
        if success:
            print("\n✅ TOKEN ACTIVATED — Added to OPD queue")
        else:
            print(f"\n❌ ACTIVATION FAILED ({reason})")

    elif choice == "3":
        served = serve_next_patient()
        if served:
            print(f"\n🩺 Serving token number {served}")
        else:
            print("\n⚠ No patients in active queue")

    elif choice == "4":
        print("\nSystem shutdown.")
        break

    else:
        print("\nInvalid option")

print("\n===== OPD CLOSED =====")
