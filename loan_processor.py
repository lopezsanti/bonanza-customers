import os
import json
import time
import glob
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LoanProcessor:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the LoanProcessor with Google GenAI credentials.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_AI_STUDIO")
        if not self.api_key:
            raise ValueError("API Key not found. Please provide it or set GOOGLE_AI_STUDIO in .env")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        self.system_instruction = """
        You are an expert data entry specialist. 
        Your task is to extract information from handwritten loan application forms.
        The forms are in Spanish.
        
        IMPORTANT: The image provided might be rotated 90 degrees clockwise or counter-clockwise. 
        Analyze the text orientation visually and extract the data correctly regardless of the image rotation.
        
        Extract data into three specific sections:
        1. SOLICITANTE
        2. CODEUDOR (First one found)
        3. CODEUDOR (Second one found, if applicable)
        
        If any information is illegible or cannot be found, leave the value blank.
        
        Return the output strictly as a JSON object.
        """

    def _save_json(self, data: dict, original_filename: str, output_folder: str):
        """Saves the extracted data to a JSON file in the specified output folder."""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_filename = Path(output_folder) / f"{Path(original_filename).stem}_extracted.json"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved: {output_filename}")

    def process_single_file(self, pdf_path: str, output_folder: Optional[str] = None) -> Optional[dict]:
        """
        Process a single PDF file and extract data.
        """
        logger.info(f"Processing: {pdf_path}")
        
        try:
            # Upload the PDF directly to Gemini
            # We use a binary stream and a safe display name to avoid UnicodeEncodeError on Windows
            with open(pdf_path, "rb") as f:
                pdf_file = self.client.files.upload(
                    file=f,
                    config=types.UploadFileConfig(
                        display_name="loan_document.pdf",
                        mime_type="application/pdf"
                    )
                )
            
            # Prompt specifically for the second page
            prompt = """
            Analyze the SECOND page of this document.
            Extract all handwritten fields from the form on that page into the defined JSON structure.
            """
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pdf_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            json_data = json.loads(response.text)
            
            # Add metadata
            json_data['meta_source_file'] = os.path.basename(pdf_path)
            json_data['meta_processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to disk if output folder is provided
            if output_folder:
                self._save_json(json_data, pdf_path, output_folder)
            
            # Cleanup: Delete the file from Gemini to avoid clutter
            self.client.files.delete(name=pdf_file.name)
            
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to extract data from {pdf_path}: {e}")
            return None

    def process_folder(self, input_folder: str, output_folder: str = "client_json_data", max_files: Optional[int] = None):
        """
        Process all PDF files in a folder.
        """
        pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in '{input_folder}'.")
            return

        if max_files:
            pdf_files = pdf_files[:max_files]
            
        logger.info(f"Selected {len(pdf_files)} files for processing.")
        
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"[{i+1}/{len(pdf_files)}] Starting {pdf_path}")
            self.process_single_file(pdf_path, output_folder)
            
            # Rate Limiting for Free Tier
            time.sleep(2)
            
        logger.info("Processing complete.")

if __name__ == "__main__":
    # Example usage when running the script directly
    processor = LoanProcessor()
    processor.process_folder("clients", "client_json_data")