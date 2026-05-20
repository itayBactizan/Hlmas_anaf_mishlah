import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np
# Import the main handler from your package
from src.models.XGB_ranker.inference_pipeline.handler_xgb import XGBRerankerHandler 


def topk_acc(topk, labels):
        return np.any(topk == labels[:, None], axis=1).mean()

class TestRealPipelineIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Runs EXACTLY ONCE before any tests start.
        Used to load the real AlephBERT model and reference tensors into RAM.
        """
        print("Initializing Real ML Models. This may take a minute...")
        
        # Create a persistent temp directory for the duration of the test suite
        cls.temp_dir = tempfile.mkdtemp()
        
        # Define paths to your actual ICBS models and dictionaries
        # Adjust these paths to where they physically live on your dev server
        cls.model_path = "/home/itayb@lamas.gov.il/my_code/models/xgb_reranker_anaf_v1/" # Or your specific fine-tuned path
        
        try:
            # Instantiate the real handler. 
            # This triggers _initialize_pipeline and loads the actual weights.
            cls.handler = XGBRerankerHandler(
                model_id=cls.model_path,
                preprocessed_data_name="/home/itayb@lamas.gov.il/Simul_AI_New_Approach/src/models/XGB_ranker/inference_pipeline/tests/test_data_small.pqt",
                temp_dir=cls.temp_dir,
                device="cuda",
                logs_dir=cls.temp_dir,
                key_column_name="input",
                prediction_column_name="prediction",
                precision_column_name="precision",
                file_suffix="real_predictions.parquet",
                simul_type="anaf", # Testing Sector (Anaf) codes
                num_threads=max(8, os.cpu_count() - 2),
                precision="int8", # Utilize the CPU quantization we built
                use_chunked=True
            )
        except Exception as e:
            cls.tearDownClass()
            raise RuntimeError(f"Failed to load real models during test setup: {e}")

    @classmethod
    def tearDownClass(cls):
        """Runs EXACTLY ONCE after all tests finish to clean up disk space."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Runs before EACH test. Used to stage fresh test data."""
        # Create a tiny batch of real-world survey responses
        # Including exact matches, ambiguous text, and a null edge case
        test_data = pd.DataFrame({
            "kod_seker": [1001, 1002, 1003, 1004, 1005],
            "input": [
                "מפתח תוכנה",           # Clean, likely an exact match or very high similarity
                "מנהל שיווק בחברת הייטק", # Complex, needs Cross-Encoder attention
                "נהג",                 # Ambiguous, needs a wide net
                "rgdfg dfg",           # Gibberish, model should still return a code safely
                None                   # Null check
            ]
        })
        
        #self.input_filename = "real_test_batch.parquet"
        #self.input_path = os.path.join(self.temp_dir, self.input_filename)
        #test_data.to_parquet(self.input_path)

    def test_handler_process_and_predict(self):
        """Validates that the adapter correctly executes the pipeline and returns a valid path."""
        
        # 1. Execute the handler exactly as the external orchestrator would
        output_file_path = self.handler.process_and_predict()
        
        # 2. Assert the handler returned a string path that actually exists
        self.assertTrue(os.path.exists(output_file_path), "The handler returned a path, but no file was written to disk.")
        
        # 3. Load the results and validate schema and counts
        results_df = pd.read_parquet(output_file_path)
        
        # 4. Check that dynamic column naming ('prediction_anaf') worked
        self.assertIn("prediction", results_df.columns)
        self.assertIn("precision", results_df.columns)
        results_df.to_csv("/home/itayb@lamas.gov.il/Simul_AI_New_Approach/src/models/XGB_ranker/inference_pipeline/tests/test-result.csv")
        # 5. Validate that predictions are actual strings and precisions are floats
        # Exclude the null/gibberish rows for this specific check if needed, 
        # but the system should map everything to at least a default float.
        #precisions = results_df["prediction_mishlah"].tolist()
        #for prec in precisions:
        #    self.assertIsInstance(prec, float, "Precision column contains non-float values.")
        data = pd.read_parquet("/home/itayb@lamas.gov.il/Simul_AI_New_Approach/src/models/XGB_ranker/inference_pipeline/tests/test_data_small.pqt")
        gt = data['SemelMishlahSofi'] if self.handler.config.simul_type == 'mishlah' else data['SemelAnafSofi']
        
        acc_3 = topk_acc(np.array(results_df['prediction'].to_list()), gt.to_numpy())
        self.assertGreaterEqual(acc_3, 0.75)
        
if __name__ == '__main__':
    unittest.main()