class Patient:
    def __init__(self, patient_id: str, base_env_id: str = "simglucose"):
        self.patient_id = patient_id
        self.safe_id = patient_id.replace("#", "_").replace(
            "/", "_"
        )  # Sanitize any special characters
        self.env_id = f"{base_env_id}-{self.safe_id}"  # For Gymnasium environment ID

    def __repr__(self):
        return f"<Patient {self.patient_id} | Env ID: {self.env_id}>"


def get_default_patients(base_env_id: str) -> list[Patient]:
    """Generate a list of Patient objects for all default profiles.

    https://github.com/jxx123/simglucose/blob/master/simglucose/params/vpatient_params.csv

    Args:
        base_env_id: The environment base ID to prefix each patient's env_id.

    Returns:
        List of Patient instances.
    """
    patients = []
    for i in range(1, 11):
        patients.append(Patient(f"adolescent#{i:03d}", base_env_id))
    for i in range(1, 11):
        patients.append(Patient(f"adult#{i:03d}", base_env_id))
    for i in range(1, 11):
        patients.append(Patient(f"child#{i:03d}", base_env_id))
    return patients
