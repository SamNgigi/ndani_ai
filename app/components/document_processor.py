import pypdf
import re
import logging
import spacy
from pathlib import Path
from typing import Dict, Union, List, Optional
from docx import Document
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a section in the document"""
    name:str
    content:str
    start_idx:int
    end_idx:int

class DocumentParser:
    """Handles parsing of different document formats with section detection"""

    def __init__(self):
        """Initialize the document parse with supported formats and load NLP model"""
        self.supported_formats = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.txt': self._parse_txt,
        }

        self.section_patterns = {
            'header': r'^(?:name|contact|personal\s+information)',
            'summary': r'^(?:summary|professional\s+summary|profile|objective)',
            'experience': r'^(?:experience|work\s+experience|employment|work\s+history)',
            'education': r'^(?:education|academic|qualifications)',
            'skills': r'^(?:skills|technical\s+skills|competencies)',
            'certifications': r'^(?:certifications|certificates|accreditations)',
            'projects': r'^(?:projects|personal\s+projects|professional\s+projects)',
            'other': r'^(?:additional|interests|volunteer|langauge|references|personal\s+milestones)',
        }

        # Load spacy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("⚠️  Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def parse(self, file_path:Union[str, Path]) -> Dict[str, str]:
        return {}

    def _parse_pdf(self, file_path:Path) -> str:
        return ""

    def _parse_docx(self, file_path:Path) -> str:
        return ""

    def _parse_txt(self, file_path:Path) -> str:
        return ""

    def _structure_content(self, text:str) -> Dict[str, str]:
        """
        Structure raw text into logical sections

        Args:
            text: Raw text extracted from document

        Returns:
            Dictionary of structured sections
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        print(cleaned_text)
        return {}

    def _detect_sections(self, text:str) -> List[Section]:
        """
        Detect document sections using regex patterns as NLP

        Args:
            text: Cleaned document text

        Returns:
            List of detected sections with their positions
        """

        sections = []
        lines = text.split("\n")
        current_position = 0

        for line in lines:
            line_stripped = line.strip().lower()
            if not line_stripped:
                current_position += len(line) + 1
                continue

            for section_name, pattern in self.section_patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Verify it's actually a header using NLP
                    if self._verify_section_header(line):
                        sections.append(
                            Section(
                                name = section_name,
                                content = '',
                                start_idx = current_position,
                                end_idx = 0 # Will be set when processing next section
                            )
                        )
                        break

                    current_position += len(line) + 1

        return sections

    def _validate_parsed_content(self):
        pass

    def _verify_section_header(self, text: str) -> bool:
        """
        Use NLP to verify if a line is likely a section header

        Args:
            text: Line ot text to verify

        Returns:
            Boolean indicating if line is likely a section header
        """
        
        doc = self.nlp(text)

        # Characteristics of section headers:
        # 1. Short length
        if len(doc) > 10:
            return False

        # 2. Usually no verbs
        has_verbs = any(token.pos_ == "VERB" for token in doc)
        if has_verbs:
            return False
        
        # 3. Often in title case or upper case
        is_title_case = bool(text.istitle() or text.isupper())

        # 4. Usually no punctuation except ':'
        has_invalid_punct = any(char for char in text if char in ',.;-()[]{}' and char != ':')
        
        return is_title_case and not has_invalid_punct
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text to clean

        Return:
            Cleaned and normalized text
        """
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n', text)

        # Removing excessive white space
        text = re.sub(r' +', ' ', text)

        # Normalize bullet points
        text = re.sub(r'[•●○◦▪▫◘◙■□▢▣▤▥▦▧▨▩]', '-', text)
        text = re.sub(r'^\s', '', text)
        # Removing page numbers i.e digit bounded by empty string at beginning and end of word
        text = re.sub(r'^\s*(?:Page\s+)?\d+[\.\s]*$', '\n', text, flags=re.MULTILINE)

        return text.strip()


