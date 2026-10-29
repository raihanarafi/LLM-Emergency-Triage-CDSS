import pandas as pd
from typing import Dict, Any, List, Tuple

class VitalsGuardrails:
    """
    Deterministic safety guardrails for vital signs to enforce high-acuity warnings.
    """
    
    # Standard Clinical Thresholds
    THRESHOLDS = {
        'o2sat': {'low_critical': 90, 'low_warning': 94},
        'sbp': {'low_critical': 90, 'high_critical': 180},
        'dbp': {'high_critical': 110},
        'heartrate': {'low_critical': 40, 'high_critical': 130},
        'resprate': {'low_critical': 8, 'high_critical': 30},
        'temperature': {'low_critical': 35.0, 'high_critical': 39.0}
    }

    @staticmethod
    def _to_float(val):
        """Safely converts string or numerical inputs to float."""
        if val is None or pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def evaluate_vitals(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Evaluates patient vitals against deterministic safety thresholds.
        
        Args:
            patient_data: Dictionary containing vitals feature-value pairs.
            
        Returns:
            Tuple[bool, List[str]]: (override_triggered, list_of_warnings)
        """
        warnings = []
        override_triggered = False

        # SpO2 / O2Sat Guardrail
        o2sat = VitalsGuardrails._to_float(
            patient_data.get('o2sat') or patient_data.get('spo2') or patient_data.get('SpO2')
        )
        if o2sat is not None:
            if o2sat < VitalsGuardrails.THRESHOLDS['o2sat']['low_critical']:
                warnings.append(f"CRITICAL: SpO2 < {VitalsGuardrails.THRESHOLDS['o2sat']['low_critical']}% ({o2sat}%) - Severe Hypoxia")
                override_triggered = True
            elif o2sat < VitalsGuardrails.THRESHOLDS['o2sat']['low_warning']:
                warnings.append(f"WARNING: Mild Hypoxia (SpO2: {o2sat}%)")

        # Systolic BP Guardrail
        sbp = VitalsGuardrails._to_float(
            patient_data.get('sbp') or patient_data.get('Systolic BP')
        )
        if sbp is not None:
            if sbp < VitalsGuardrails.THRESHOLDS['sbp']['low_critical']:
                warnings.append(f"CRITICAL: Systolic BP < {VitalsGuardrails.THRESHOLDS['sbp']['low_critical']} mmHg ({sbp} mmHg) - Hypotension")
                override_triggered = True
            elif sbp > VitalsGuardrails.THRESHOLDS['sbp']['high_critical']:
                warnings.append(f"CRITICAL: Systolic BP > {VitalsGuardrails.THRESHOLDS['sbp']['high_critical']} mmHg ({sbp} mmHg) - Hypertensive Crisis")
                override_triggered = True

        # Heart Rate Guardrail
        hr = VitalsGuardrails._to_float(
            patient_data.get('heartrate') or patient_data.get('hr') or patient_data.get('Heart Rate')
        )
        if hr is not None:
            if hr < VitalsGuardrails.THRESHOLDS['heartrate']['low_critical']:
                warnings.append(f"CRITICAL: Severe Bradycardia (HR: {hr} bpm)")
                override_triggered = True
            elif hr > VitalsGuardrails.THRESHOLDS['heartrate']['high_critical']:
                warnings.append(f"WARNING: Severe Tachycardia (HR: {hr} bpm)")
                override_triggered = True

        # Respiratory Rate Guardrail
        resp = VitalsGuardrails._to_float(
            patient_data.get('resprate') or patient_data.get('resp_rate') or patient_data.get('Resp Rate')
        )
        if resp is not None:
            if resp < VitalsGuardrails.THRESHOLDS['resprate']['low_critical'] or resp > VitalsGuardrails.THRESHOLDS['resprate']['high_critical']:
                warnings.append(f"CRITICAL: Abnormal Respiratory Rate ({resp} breaths/min)")
                override_triggered = True

        return override_triggered, warnings
