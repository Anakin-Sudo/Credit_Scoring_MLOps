import pandas as pd


def curate_german_credit(input_path: str, output_path: str):
    """
    Convert raw UCI German Credit .DATA file into a clean CSV with proper column names,
    decoded categorical variables, and binary target (0=good, 1=bad).
    """

    # Column names from UCI dataset description
    columns = [
        'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
        'Savings', 'Employment', 'InstallmentRate', 'SexAndStatus',
        'OtherDetors', 'ResidenceSince', 'Property', 'Age',
        'OtherInstallmentPlans', 'Housing', 'ExistingCredits',
        'Job', 'PeopleLiable', 'Telephone', 'ForeignWorker', 'CreditRisk'
    ]

    # Load raw .DATA file
    df = pd.read_csv(input_path,
                     delim_whitespace=True,
                     header=None,
                     names=columns)

    # --- Decoding dictionaries (from UCI documentation) ---
    status_map = {
        "A11": "< 0 DM",
        "A12": "0 ≤ balance < 200 DM",
        "A13": "≥ 200 DM",
        "A14": "no checking account"
    }

    credit_history_map = {
        "A30": "no credits taken",
        "A31": "all credits paid back duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account/other credits existing"
    }

    purpose_map = {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/TV",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others"
    }

    savings_map = {
        "A61": "< 100 DM",
        "A62": "100 ≤ ... < 500 DM",
        "A63": "500 ≤ ... < 1000 DM",
        "A64": "≥ 1000 DM",
        "A65": "unknown/none"
    }

    employment_map = {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 ≤ ... < 4 years",
        "A74": "4 ≤ ... < 7 years",
        "A75": "≥ 7 years"
    }

    sex_status_map = {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single"
    }

    other_debtors_map = {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor"
    }

    property_map = {
        "A121": "real estate",
        "A122": "building society savings/life insurance",
        "A123": "car or other",
        "A124": "unknown/none"
    }

    other_installment_map = {
        "A141": "bank",
        "A142": "stores",
        "A143": "none"
    }

    housing_map = {
        "A151": "rent",
        "A152": "own",
        "A153": "for free"
    }

    job_map = {
        "A171": "unemployed/unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified"
    }

    telephone_map = {
        "A191": "none",
        "A192": "yes, registered under customer’s name"
    }

    foreign_worker_map = {
        "A201": "yes",
        "A202": "no"
    }

    # --- Apply decoding ---
    df["Status"] = df["Status"].map(status_map)
    df["CreditHistory"] = df["CreditHistory"].map(credit_history_map)
    df["Purpose"] = df["Purpose"].map(purpose_map)
    df["Savings"] = df["Savings"].map(savings_map)
    df["Employment"] = df["Employment"].map(employment_map)
    df["SexAndStatus"] = df["SexAndStatus"].map(sex_status_map)
    df["OtherDetors"] = df["OtherDetors"].map(other_debtors_map)
    df["Property"] = df["Property"].map(property_map)
    df["OtherInstallmentPlans"] = df["OtherInstallmentPlans"].map(other_installment_map)
    df["Housing"] = df["Housing"].map(housing_map)
    df["Job"] = df["Job"].map(job_map)
    df["Telephone"] = df["Telephone"].map(telephone_map)
    df["ForeignWorker"] = df["ForeignWorker"].map(foreign_worker_map)

    # --- Recode target ---
    # 1 = good credit -> 0 (non-default), 2 = bad credit -> 1 (default)
    df["CreditRisk"] = df["CreditRisk"].map({1: 0, 2: 1})

    # Save curated dataset
    df.to_csv(output_path, index=False)
    print(f"Curated dataset saved to {output_path} with shape {df.shape}")


if __name__ == "__main__":
    # Example usage
    curate_german_credit(
        input_path="german.data",   # your .DATA file
        output_path="german_credit.csv"         # clean CSV with decoded categories
    )
