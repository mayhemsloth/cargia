"""
Text cleaner module for cleaning thought text using Gemma3.
"""
import sqlite3
import os
from typing import List, Dict, Optional, Callable
from cargia.common.gemma3_wrapper import Gemma3Wrapper
from cargia.data_manager import get_repo_root, log_error


class TextCleaner:
    """
    A class to handle cleaning of thought text using Gemma3.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the text cleaner.
        
        Args:
            data_dir: Path to the data directory containing the thoughts database
        """
        self.data_dir = os.path.abspath(data_dir)
        self.thoughts_db_path = os.path.join(self.data_dir, "thoughts.db")
        self.gemma_wrapper = None
        
    def initialize_gemma(self, model_id: str = "google/gemma-3-4b-it"):
        """
        Initialize the Gemma3 wrapper.
        
        Args:
            model_id: Hugging Face model ID to use for cleaning
        """
        if self.gemma_wrapper is None:
            self.gemma_wrapper = Gemma3Wrapper(model_id=model_id, max_new_tokens=1024)
            self.gemma_wrapper.initialize()
    
    def get_thoughts_needing_cleaning(self) -> List[Dict]:
        """
        Get all thoughts that need cleaning (have thought_text but no cleaned_thought_text).
        
        Returns:
            List of thought dictionaries with id, thought_text, and other fields
        """
        if not os.path.exists(self.thoughts_db_path):
            return []
        
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, solve_id, pair_label, pair_type, sequence_index, thought_text
                FROM thoughts
                WHERE thought_text IS NOT NULL 
                AND thought_text != ''
                AND (cleaned_thought_text IS NULL OR cleaned_thought_text = '')
            """)
            
            thoughts = []
            for row in cursor.fetchall():
                thoughts.append({
                    'id': row[0],
                    'solve_id': row[1],
                    'pair_label': row[2],
                    'pair_type': row[3],
                    'sequence_index': row[4],
                    'thought_text': row[5]
                })
            
            return thoughts
        finally:
            conn.close()
    
    def get_all_thoughts_with_text(self) -> List[Dict]:
        """
        Get all thoughts that have text (regardless of cleaning status).
        
        Returns:
            List of thought dictionaries with id, thought_text, and other fields
        """
        if not os.path.exists(self.thoughts_db_path):
            return []
        
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, solve_id, pair_label, pair_type, sequence_index, thought_text
                FROM thoughts
                WHERE thought_text IS NOT NULL 
                AND thought_text != ''
            """)
            
            thoughts = []
            for row in cursor.fetchall():
                thoughts.append({
                    'id': row[0],
                    'solve_id': row[1],
                    'pair_label': row[2],
                    'pair_type': row[3],
                    'sequence_index': row[4],
                    'thought_text': row[5]
                })
            
            return thoughts
        finally:
            conn.close()
    
    def clean_single_thought(self, thought_text: str) -> str:
        """
        Clean a single thought text using Gemma3.
        
        Args:
            thought_text: The raw thought text to clean
            
        Returns:
            The cleaned thought text
        """
        if not self.gemma_wrapper:
            raise RuntimeError("Gemma3 wrapper not initialized. Call initialize_gemma() first.")
        
        try:
            return self.gemma_wrapper.clean_thought_text(thought_text)
        except ImportError as e:
            if "libtriton" in str(e) or "DLL" in str(e):
                log_error(f"Triton DLL error while cleaning thought text: {thought_text[:100]}...", e)
                print(f"Warning: Triton DLL error detected. This is a known Windows issue. Returning original text.")
                return thought_text
            else:
                log_error(f"Import error while cleaning thought text: {thought_text[:100]}...", e)
                return thought_text
        except Exception as e:
            log_error(f"Failed to clean thought text: {thought_text[:100]}...", e)
            # Return original text if cleaning fails
            return thought_text
    
    def update_cleaned_thought(self, thought_id: int, cleaned_text: str):
        """
        Update a thought record with cleaned text.
        
        Args:
            thought_id: The ID of the thought to update
            cleaned_text: The cleaned text to store
        """
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE thoughts
                SET cleaned_thought_text = ?
                WHERE id = ?
            """, (cleaned_text, thought_id))
            conn.commit()
        finally:
            conn.close()
    
    def clean_all_thoughts(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, int]:
        """
        Clean all thoughts that need cleaning.
        
        Args:
            progress_callback: Optional callback function(processed, total) for progress updates
            
        Returns:
            Dictionary with statistics about the cleaning process
        """
        thoughts_to_clean = self.get_thoughts_needing_cleaning()
        total_thoughts = len(thoughts_to_clean)
        
        if total_thoughts == 0:
            return {
                'total_thoughts': 0,
                'cleaned_thoughts': 0,
                'failed_thoughts': 0
            }
        
        print(f"Found {total_thoughts} thoughts that need cleaning.")
        
        cleaned_count = 0
        failed_count = 0
        
        for i, thought in enumerate(thoughts_to_clean):
            try:
                # Clean the thought text
                cleaned_text = self.clean_single_thought(thought['thought_text'])
                
                # Update the database
                self.update_cleaned_thought(thought['id'], cleaned_text)
                
                cleaned_count += 1
                print(f"Cleaned thought {i+1}/{total_thoughts} (ID: {thought['id']})")
                
            except Exception as e:
                failed_count += 1
                log_error(f"Failed to clean thought {thought['id']}", e)
                print(f"Failed to clean thought {i+1}/{total_thoughts} (ID: {thought['id']}): {str(e)}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_thoughts)
        
        print(f"Text cleaning completed. Cleaned: {cleaned_count}, Failed: {failed_count}")
        
        return {
            'total_thoughts': total_thoughts,
            'cleaned_thoughts': cleaned_count,
            'failed_thoughts': failed_count
        }
    
    def get_cleaning_stats(self) -> Dict[str, int]:
        """
        Get statistics about the cleaning status of thoughts.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if not os.path.exists(self.thoughts_db_path):
            return {
                'total_thoughts': 0,
                'cleaned_thoughts': 0,
                'uncleaned_thoughts': 0
            }
        
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            # Get total thoughts with text
            cursor.execute("""
                SELECT COUNT(*) FROM thoughts 
                WHERE thought_text IS NOT NULL AND thought_text != ''
            """)
            total_thoughts = cursor.fetchone()[0]
            
            # Get cleaned thoughts
            cursor.execute("""
                SELECT COUNT(*) FROM thoughts 
                WHERE cleaned_thought_text IS NOT NULL AND cleaned_thought_text != ''
            """)
            cleaned_thoughts = cursor.fetchone()[0]
            
            uncleaned_thoughts = total_thoughts - cleaned_thoughts
            
            return {
                'total_thoughts': total_thoughts,
                'cleaned_thoughts': cleaned_thoughts,
                'uncleaned_thoughts': uncleaned_thoughts
            }
        finally:
            conn.close()
    
    def clean_all_thoughts_overwrite(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, int]:
        """
        Clean all thoughts that have text, overwriting any existing cleaned text.
        
        Args:
            progress_callback: Optional callback function(processed, total) for progress updates
            
        Returns:
            Dictionary with statistics about the cleaning process
        """
        thoughts_to_clean = self.get_all_thoughts_with_text()
        total_thoughts = len(thoughts_to_clean)
        
        if total_thoughts == 0:
            return {
                'total_thoughts': 0,
                'cleaned_thoughts': 0,
                'failed_thoughts': 0
            }
        
        print(f"Found {total_thoughts} thoughts to clean (overwrite mode).")
        
        cleaned_count = 0
        failed_count = 0
        
        for i, thought in enumerate(thoughts_to_clean):
            try:
                # Clean the thought text
                cleaned_text = self.clean_single_thought(thought['thought_text'])
                
                # Update the database
                self.update_cleaned_thought(thought['id'], cleaned_text)
                
                cleaned_count += 1
                print(f"Cleaned thought {i+1}/{total_thoughts} (ID: {thought['id']})")
                
            except Exception as e:
                failed_count += 1
                log_error(f"Failed to clean thought {thought['id']}", e)
                print(f"Failed to clean thought {i+1}/{total_thoughts} (ID: {thought['id']}): {str(e)}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_thoughts)
        
        print(f"Text cleaning completed (overwrite mode). Cleaned: {cleaned_count}, Failed: {failed_count}")
        
        return {
            'total_thoughts': total_thoughts,
            'cleaned_thoughts': cleaned_count,
            'failed_thoughts': failed_count
        } 