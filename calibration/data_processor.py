import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
TRAC EOIR DATA MAPPINGS
"""

# Asylum type mapping
ASY_TYPE_MAP = {
    'E': 'defensive',   # Defensive: before immigration judge
    'I': 'affirmative', # Affirmative: with DHS asylum office
}

# Relief decision mapping
RELIEF_DECISION_MAP = {
    'G': 'granted', 'C': 'granted', 'F': 'granted', 'I': 'granted',
    'D': 'denied',
    'O': 'other_exit', 'W': 'other_exit', 'A': 'other_exit'
}

# IJ Decision Code mapping — dec_code is the IJ's Procedural decision.
DEC_CODE_MAP = {
    # Grants
    'A': 'granted', 'G': 'granted', 'W': 'granted',
    # Denials / Removal orders
    'R': 'denied', 'D': 'denied',
    # Procedural / Administrative exits
    'X': 'other_exit', 'T': 'other_exit', 'V': 'other_exit',
    'E': 'other_exit', 'O': 'other_exit', 'Z': 'other_exit',
    'U': 'other_exit',  # Admin close/undefined proceeding
    'L': 'other_exit',  # Legacy LPR adjustment or leave-to-depart
    'J': 'other_exit',  # Jurisdiction return/legacy code
    'H': 'other_exit',  # Legacy humanitarian hold
    'S': 'other_exit',  # Stipulated removal or sustained
    'C': 'other_exit',  # Certified to BIA
}

# Custody mapping
CUSTODY_MAP = {
    'N': 'never_detained',
    'R': 'released',
    'D': 'detained',
}

# ============================================
# DATA PROCESSOR
# ============================================

class CompleteTRACDataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
        self.hearing_loc_map = None

    def load_all_data(self):
        print("=" * 70)
        print("COMPLETE TRAC EOIR DATA PROCESSING")
        print("=" * 70)

        print("\n1. Loading master data...")
        master = self._load_master_data()
        print(f"   Loaded {len(master):,} proceeding records")

        print("\n2. Loading relief applications...")
        relief = self._load_relief_data()

        print("\n3. Loading hearing location lookup...")
        self._load_hearing_locations()

        print("\n4. Mapping codes to meaningful values...")
        master = self._map_all_codes(master)

        print("\n5. Merging relief outcomes...")
        df = self._merge_relief_outcomes(master, relief)

        print("\n6. Cleaning dates...")
        df = self._clean_dates(df)

        print("\n7. Creating derived attributes...")
        df = self._add_derived_attributes(df)

        print("\n8. Filtering to valid ASYLUM records...")
        df = self._filter_valid_records(df)

        print("\n9. Aggregating to case level...")
        df = self._aggregate_to_case_level(df)

        print("\n10. Clustering courts into 4 Simulation Archetypes...")
        df = self._cluster_courts(df)

        self.df = df
        self._print_summary_stats()
        return self.df

    def _load_master_data(self) -> pd.DataFrame:
        useful_cols = [
            "idncase", "idnproceeding", "cosc_date", "cinput_date", "ccomp_date",
            "case_type", "c_asy_type", "base_city_code", "hearing_loc_code",
            "ij_code", "custody", "attorney_flag", "absentia", "dec_code",
            "nbr_of_appeals", "nbr_of_charges", "crim_ind", "nat", "gender", "lang"
        ]
        master = pd.read_csv(
            f"{self.data_dir}/master.csv",
            usecols=lambda col: col in useful_cols,
            dtype=str,
            low_memory=False,
            encoding='latin-1'
        )
        for col in master.columns:
            if master[col].dtype == 'object':
                master[col] = master[col].str.strip().str.upper()
        for col in ['nbr_of_appeals', 'nbr_of_charges']:
            if col in master.columns:
                master[col] = pd.to_numeric(master[col], errors='coerce')
        return master

    def _load_relief_data(self) -> pd.DataFrame:
        try:
            relief = pd.read_csv(
                f"{self.data_dir}/reliefApplications.csv",
                dtype=str,
                encoding='latin-1'
            )
        except FileNotFoundError:
            print("   Relief file not found - continuing without")
            return pd.DataFrame()

        if 'appl_code' in relief.columns:
            relief['appl_code'] = relief['appl_code'].str.strip().str.upper()
        if 'appl_dec' in relief.columns:
            relief['appl_dec'] = relief['appl_dec'].str.strip().str.upper()

        relief_asylum = (
            relief[relief['appl_code'] == 'ASYL'].copy()
            if 'appl_code' in relief.columns else relief
        )
        print(f"   Total relief records: {len(relief):,}")
        print(f"   Asylum relief records: {len(relief_asylum):,}")

        if len(relief_asylum) == 0:
            return pd.DataFrame()

        relief_asylum['outcome'] = relief_asylum['appl_dec'].map(RELIEF_DECISION_MAP)
        relief_asylum['relief_granted'] = relief_asylum['appl_dec'].isin(['G', 'C'])
        return relief_asylum

    def _load_hearing_locations(self):
        try:
            hloc = pd.read_csv(f"{self.data_dir}/hearing_loc_lookup.csv", encoding='latin-1')
            hloc.columns = hloc.columns.str.replace('ï»¿', '').str.strip()
            self.hearing_loc_map = dict(zip(
                hloc['Code'].str.strip().str.upper(),
                hloc['State'].str.strip()
            ))
            print(f"   Loaded {len(self.hearing_loc_map)} hearing locations")
        except FileNotFoundError:
            print("   Hearing location file not found")
            self.hearing_loc_map = {}

    def _map_all_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        df['asylum_type'] = df['c_asy_type'].map(ASY_TYPE_MAP)
        df['is_affirmative'] = df['asylum_type'] == 'affirmative'
        df['is_defensive'] = df['asylum_type'] == 'defensive'
        df['custody_status'] = df['custody'].map(CUSTODY_MAP)
        df['decision_category'] = df['dec_code'].map(DEC_CODE_MAP)

        if self.hearing_loc_map:
            df['state'] = df['hearing_loc_code'].map(self.hearing_loc_map)
            missing_mask = df['state'].isna() & df['hearing_loc_code'].notna()
            unknown = df.loc[missing_mask, 'hearing_loc_code'].unique()
            if len(unknown) > 0:
                print(f"   WARNING: {len(unknown)} hearing_loc_codes not in lookup")
        return df

    def _merge_relief_outcomes(self, master: pd.DataFrame, relief: pd.DataFrame) -> pd.DataFrame:
        if relief.empty:
            print("   No relief data to merge")
            master['relief_granted'] = False
            master['asylum_granted'] = False
            master['outcome'] = None
            return master

        relief_agg = relief.groupby('idnproceeding').agg({
            'relief_granted': 'max',
            'outcome': lambda x: x.iloc[0] if len(x) > 0 else None,
        }).reset_index()

        df = master.merge(relief_agg, on='idnproceeding', how='left')
        df['relief_granted'] = df['relief_granted'].fillna(False).astype(bool)
        df['asylum_granted'] = df['relief_granted']

        n_granted = df['asylum_granted'].sum()
        print(f"   Protection granted: {n_granted:,} proceedings "
              f"({n_granted/len(df):.3f} of all asylum proceedings)")
        return df

    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['cosc_date', 'cinput_date', 'ccomp_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df['arrival_date'] = df['cinput_date']
        df['completion_date'] = df['ccomp_date']

        df['arrival_year'] = df['arrival_date'].dt.year
        df['arrival_month'] = df['arrival_date'].dt.month
        df['completion_year'] = df['completion_date'].dt.year

        df['processing_days'] = (df['completion_date'] - df['arrival_date']).dt.days
        df['processing_months'] = (df['processing_days'] / 30.44).round(2)
        df['processing_months'] = df['processing_months'].clip(lower=0, upper=360)

        # is_pending at proceeding level — re-derived after aggregation
        df['is_pending'] = df['completion_date'].isna()
        return df

    def _add_derived_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        df['has_representation'] = df['attorney_flag'] == '1'
        df['is_detained'] = df['custody'] == 'D'
        df['is_released'] = df['custody'] == 'R'
        df['missed_hearing'] = df['absentia'] == 'Y'
        df['has_appeal'] = df['nbr_of_appeals'] > 0
        df['has_criminal_record'] = df['crim_ind'] == 'Y'
        return df

    def _filter_valid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df[(df['arrival_year'] >= 1990) & (df['arrival_year'] <= 2025)]
        df = df[df['idnproceeding'].notna()]
        df = df[df['base_city_code'].notna()]
        # Require explicit asylum type OR a relief application record
        is_asylum_case = df['is_affirmative'] | df['is_defensive'] | df['outcome'].notna()
        df = df[is_asylum_case]
        print(f"   Filtered from {initial_count:,} to {len(df):,} valid ASYLUM records")
        return df

    def _aggregate_to_case_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate from proceeding level to case level.

        asylum_granted uses 'max' — True if ANY proceeding resulted in a grant.
        """
        df_chron = df.sort_values(['idncase', 'arrival_date', 'completion_date'])

        case_agg = df_chron.groupby('idncase').agg(
            idnproceeding      =('idnproceeding',       'last'),
            arrival_date       =('arrival_date',         'first'),
            completion_date    =('completion_date',      'last'),
            processing_months  =('processing_months',    'last'),
            asylum_granted     =('asylum_granted',       'max'),   # True if ever granted
            relief_granted     =('relief_granted',       'max'),
            has_representation =('has_representation',   'first'),
            is_detained        =('is_detained',          'first'),
            is_released        =('is_released',          'first'),
            missed_hearing     =('missed_hearing',       'max'),
            has_appeal         =('has_appeal',           'max'),
            has_criminal_record=('has_criminal_record',  'first'),
            is_affirmative     =('is_affirmative',       'first'),
            is_defensive       =('is_defensive',         'first'),
            base_city_code     =('base_city_code',       'first'),
            hearing_loc_code   =('hearing_loc_code',     'first'),
            ij_code            =('ij_code',              'last'),
            state              =('state',                'first'),
            custody_status     =('custody_status',       'first'),
            nbr_of_charges     =('nbr_of_charges',       'first'),
            nat                =('nat',                  'first'),
            gender             =('gender',               'first'),
        ).reset_index()

        # Best IJ procedural decision — used for queue routing, not grant rate
        PRIORITY_ORDER = ['granted', 'denied', 'other_exit']

        def best_decision(decisions):
            vals = set(decisions.dropna())
            for candidate in PRIORITY_ORDER:
                if candidate in vals:
                    return candidate
            return None

        best_dec = df_chron.groupby('idncase')['decision_category'].apply(best_decision)
        case_agg = case_agg.merge(best_dec.rename('decision_category'), on='idncase', how='left')

        case_agg['is_pending'] = case_agg['completion_date'].isna()

        print(f"   Aggregated from {len(df):,} proceedings to {len(case_agg):,} cases")
        print("\n   --- Decision category distribution (case level) ---")
        print(case_agg['decision_category'].value_counts(dropna=False).to_string())
        print(f"\n   Overall grant rate (relief): {case_agg['asylum_granted'].mean():.3f}")

        return case_agg

    def _cluster_courts(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cluster on relief_granted grant rate — the authoritative protection signal.
        court_stats = df.groupby('base_city_code').agg(
            total_cases=('idncase', 'count'),
            grant_rate=('asylum_granted', 'mean'),
        ).reset_index()

        valid_courts = court_stats[court_stats['total_cases'] > 100].copy()

        scaler = StandardScaler()
        features = scaler.fit_transform(valid_courts[['total_cases', 'grant_rate']])
        kmeans = KMeans(n_clusters=4, random_state=42)
        valid_courts['court_cluster'] = kmeans.fit_predict(features)

        df = df.merge(valid_courts[['base_city_code', 'court_cluster']], on='base_city_code', how='left')
        df['court_cluster'] = df['court_cluster'].fillna(-1).astype(int)

        print("\n   --- Court Cluster Archetypes (relief grant rate) ---")
        cluster_summary = valid_courts.groupby('court_cluster').agg(
            num_courts=('base_city_code', 'count'),
            avg_volume=('total_cases', 'mean'),
            avg_grant_rate=('grant_rate', 'mean')
        ).round(3)
        print(cluster_summary)
        return df

    def _print_summary_stats(self):
        print("\n" + "=" * 70)
        print("DATA QUALITY REPORT")
        print("=" * 70)

        print(f"\n--- Basic Counts ---")
        print(f"Total asylum cases: {len(self.df):,}")
        print(f"Pending:   {self.df['is_pending'].sum():,}")
        print(f"Completed: {(~self.df['is_pending']).sum():,}")

        completed = self.df[~self.df['is_pending']]

        print(f"\n--- Protection Outcomes ---")
        print(f"Asylum/protection granted: {self.df['asylum_granted'].sum():,}")
        print(f"Overall grant rate:        {self.df['asylum_granted'].mean():.3f}")
        print(f"(Benchmark: ~0.29 nationally for defensive cases 2019, ~0.09 all case types)")

        print(f"\n--- Case Attributes ---")
        print(f"Has representation: {self.df['has_representation'].mean():.3f}")
        print(f"Detained:           {self.df['is_detained'].mean():.3f}")
        print(f"Missed hearing:     {self.df['missed_hearing'].mean():.3f}")
        print(f"Has criminal record:{self.df['has_criminal_record'].mean():.3f}")

        print(f"\n--- Appeal Rates (denied dec_code cases only) ---")
        denied_cases = self.df[self.df['decision_category'] == 'denied']
        if not denied_cases.empty:
            appeal_rates = denied_cases.groupby('has_representation')['has_appeal'].mean()
            print(f"  Unrepresented: {appeal_rates.get(False, 0):.3f}")
            print(f"  Represented:   {appeal_rates.get(True, 0):.3f}")

        print(f"\n--- Processing Time / months (completed cases) ---")
        print(f"  Mean:     {completed['processing_months'].mean():.1f}")
        print(f"  Median:   {completed['processing_months'].median():.1f}")
        print(f"  95th pct: {completed['processing_months'].quantile(0.95):.1f}")

        print(f"\n--- Top 10 Courts by Volume ---")
        print(self.df['base_city_code'].value_counts().head(10).to_string())


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    processor = CompleteTRACDataProcessor(data_dir="../data")
    df = processor.load_all_data()
    output_path = "../data/mock_data_FINAL.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")
    return df

if __name__ == "__main__":
    df = main()