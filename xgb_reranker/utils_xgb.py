
import pandas as pd 
import re 

COLS_TO_TEXT = {
            "TeurAnafMale": "שם חברה",
            "TeurPeula": "פרוט פעילויות בעבודה",
            "EzoAvoda": "עבודה עיקרית",
            "Gil": "גיל",
            "gil": "גיל",
            "ShemAvoda": "שם מקום עבודה",
            "SugAvoda": "פעילות עיקרית של מקום העבודה",
            "MaamadAvoda": "מעמד עבודה",
            "ShemMachlaka": "שם מחלקה",
            "SugMachlaka": "פעילות עיקרית של מחלקה",
            "TeudaGvoha": "תעודה",
            "TeudaGvohaAcher": "תעודה אחרת",
            "MakorSachar": "מקור שכר",
            "TeurTafkid": "תאור תפקיד",
            "TeurMishlahMale": "תאור עיסוקים",
            "shnotlimud": "שנות לימוד",
            "MenaheletMi": "מנהל",
            "MenahelEtMi": "מנהל",
            "YeshuvAvoda": "יישוב עבודה",
            "unknown": "לא ידוע"
        }

MAAMAD_DICT = {
    "1": "שכיר", "2": "עצמאי המעסיק אחד עד שניים שכירים", "3": "עצמאי המעסיק שלושה שכירים ויותר",
    "4": "עצמאי שאינו מעסיק שכירים", "5": "חבר קואופרטיב", "6": "חבר קיבוץ",
    "7": "בן משפחה העובד ללא תשלום", "8": "שכיר בעל חברה בעמ או עסק", "9": "עצמאי",
    "10": "עצמאי: בעל עסק או מקבל תשלום מלקוחות", "11": "מנהל חברה בעמ בבעלותך או בשליטתך (לפחות 51% שליטה)",
    "12": "שכיר כולל חבר קאופרטיב", "98": "לא ידוע", "99": "לא ידוע", "": "לא ידוע"
}

TEUDA_DICT = {
    "1": "תעודת סיום של בית ספר יסודי או חטיבת ביניים", "2": "תעודת סיום תיכון (שאיננה תעודת בגרות)",
    "3": "תעודת בגרות", "4": "תעודה של בית ספר על-תיכוני שאינה תעודה אקדמית",
    "5": "תואר אקדמי ראשון, B.A, או תואר מקביל", "6": "תואר אקדמי שני, M.A, או תואר מקביל",
    "7": "תואר אקדמי שלישי, PH.D, או תואר מקביל", "8": "תעודה אחרת", "9": "לא קיבל אף תעודה",
    "98": "לא ידוע", "99": "לא ידוע", "": "לא ידוע"
}

SACHAR_DICT = {
    "1": "מקום העבודה", "2": "חברת כח אדם", "3": "אחר",
    "4": "מחברת קבלן, בית תכנה, חברת שמירה, נקיון וכדומה"
}

MENAHEL_DICT = {"1": "עובדים", "2": "מנהלים", "3": "מנהלים ועובדים"}

def get_yeshuv_dict(path="/home/nfsdisk1/Simul_AI/other_data_Amir/yishuv_dict.csv"):
        df_yishuv = pd.read_csv(path, encoding="utf-8-sig")
        df_yishuv.rename({"1314": "code", "נווה זיו": "name"}, inplace=True)
        d = df_yishuv.to_dict(orient="index")
        dict_yishuv = {}
        for i, v in d.items():
            dict_yishuv[str(v["1314"])] = v[ "נווה זיו"]
        dict_yishuv["1314"] = "נווה זיו"
        return dict_yishuv

COLUMN_HANDLERS = {
    "MaamadAvoda": MAAMAD_DICT,
    "MakorSachar": SACHAR_DICT,
    "MenaheletMi": MENAHEL_DICT,
    "MenahelEtMi": MENAHEL_DICT,
    "TeudaGvoha": TEUDA_DICT
}


def _serialize_data_vectorized(df: pd.DataFrame,  yishuv_path: str="/home/nfsdisk1/Simul_AI/other_data_Amir/yishuv_dict.csv", use_e5: bool = False,) -> pd.Series:
        """
        High-performance vectorized serialization using global mapping dictionaries.
        """
        subset = ['TeurAnafMale', 'TeurMishlahMale', 'TeurTafkid', 'MaamadAvoda', 'ShemAvoda', 'SugAvoda', 'SugMachlaka', 'EzoAvoda', 'TeurPeula',
                  'MakorSachar', 'ShemMachlaka', 'MenaheletMi', 'MenahelEtMi' ,"Gil", "TeudaGvoha", "YeshuvAvoda", "shnotlimud"]
        
        COLUMN_HANDLERS["YeshuvAvoda"] = get_yeshuv_dict(yishuv_path)
        # Filter to only existing columns to avoid KeyErrors
        valid_subset = ['ShemAvoda', 'SugAvoda', 'ShemMachlaka', 'SugMachlaka', 'EzoAvoda', 'TeurPeula', 'TeurTafkid', 'MaamadAvoda', 'MakorSachar', 'MenahelEtMi', 'MenaheletMi', "Gil", 'gil', "TeudaGvoha", "YeshuvAvoda", "shnotlimud"]
        
        # Work on a copy converting to string
        work_df = df[[col for col in subset if col in df.columns]].astype(str).copy()
        unknown_text = COLS_TO_TEXT['unknown']
        if "MenahelEtMi" in work_df.columns:
             work_df.rename({'MenahelEtMi': "MenaheletMi"})
        if "gil" in work_df.columns:
             work_df.rename({'gil': "Gil"}) 
        valid_subset = [c for c in valid_subset if c in work_df.columns]
        forbidden_words = ['none', 'n/a', 'unknown', 'null', 'nan', 'לא ידוע', 'אין', 'חסר']
        forbidden_pattern = r'^(?:' + '|'.join(map(re.escape, forbidden_words)) + r')$'


        # Cleaning & Mapping
        for col in valid_subset:
            # Strip whitespace and clean forbidden words
            work_df[col] = work_df[col].str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(r'\.0$', '', regex=True)
            is_forbidden = work_df[col].str.contains(forbidden_pattern, case=False, na=False, regex=True)
            work_df[col] = work_df[col].mask(is_forbidden,"")

            # Map codes to text
            if col in COLUMN_HANDLERS:
                work_df[col] = work_df[col].map(COLUMN_HANDLERS[col]).fillna(unknown_text)
            
            # Clean formatting (.0)
            elif col in ["gil", "shnotlimud", "Gil"]:
                #work_df[col] = work_df[col]
                work_df[col] = work_df[col].replace({'nan': unknown_text, 'None': unknown_text, '<NA>': unknown_text})
            else:
                work_df[col] = work_df[col].str.replace(r'[^0-9A-Za-z\u0590-\u05FF ]+', "", regex=True)
                is_only_digit = work_df[col].str.match(r'^\s*\d+\s*$', na=False) 
                work_df[col] = work_df[col].mask(is_only_digit, "")

        series_list = []
        for col in valid_subset:
            prefix = COLS_TO_TEXT.get(col, col)
            val = work_df[col]
            part = prefix + ": " + val
            series_list.append(part.mask(val == "", ""))

        if not series_list:
            return pd.Series("", index=df.index)

        # Fast Join
        final_series = series_list[0]
        for s in series_list[1:]:
            final_series = final_series + "; " + s

        
        final_series = [
             "; ".join([p.strip() for p in s.split(';') if p.split()]) for s in final_series 
             ]
        
        #final_series = final_series + "."
        if use_e5:
            final_series = "query: " + final_series
            
        return final_series