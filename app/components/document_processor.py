from ..verify_ollama import get_error_details
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
        """
        Main parsing interface that handles different file formats

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing structured document section
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")

        try:
            # Extract raw text using format-specific parser
            raw_text = self.supported_formats[file_ext](file_path)

            # Process and structure the text
            structured_content = self._structure_content(raw_text)

            # Validate the parsed content
            self._validate_parsed_content(structured_content)

            return structured_content

        except Exception as e:
            logger.error(f"❌ Error parsing documents: {get_error_details(e)}")
            raise


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

        # Detect sections
        sections = self._detect_sections(cleaned_text)

        #Initialize structured content dict with empty sections
        structured_content = {
            "header": '',
            "summary": '',
            "experience": '',
            "education": '',
            "skills": '',
            "certifications": '',
            "projects": '',
            "other": '',
        }

        sorted_sections = sorted(sections, key=lambda x:x.start_idx)

        for i, section in enumerate(sorted_sections):
            # Determine end of section
            if i < len(sorted_sections) - 1:
                content = cleaned_text[section.start_idx:sorted_sections[i+1].start_idx].strip()
            else:
                content = cleaned_text[section.start_idx:].strip()

            # Remove section header from content
            content = self._remove_section_header(content)

            # Store in appropriate section
            if section.name in structured_content:
                structured_content[section.name] = content
            else:
                structured_content['other'] += f"\n{content}" if structured_content['other'] else content



        return structured_content

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

    def _validate_parsed_content(self, content:Dict[str, str]) -> None:
        """
        Validate parsed content structure and content

        Args:
            content: Structured content dictionary to validate

        Raises:
            ValueError: If content structure is invalid
        """
        # Check for required sections
        required_sections = {'header','summary','experience','education','skills'}
        missing_sections = required_sections - set(content.keys())
        if missing_sections:
            logger.warning(f"⚠️  Missing required sections: {missing_sections}")

        # Check for empty sections
        empty_sections = [section for section, text in content.items() if not text.strip()]
        if empty_sections:
            logger.warning(f"⚠️  Empty sections detected: {empty_sections}")

        # Basic content validation
        for section, text in content.items():
            if not isinstance(text, str):
                raise ValueError(f"Invalid content type in section '{section}': expected str, got {type(text)}")

            # Check for potential parseing errors (e.g, garbage characters)
            if re.search(r'[^\100-\x7f]+', text):
                logger.warning(f"⚠️  Non-ACII characters detected in section '{section}'")


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

    def _remove_section_header(self, text:str) -> str:
        """
        Remove section header from section content

        Args:
            text: Section text including header

        Returns:
            Section content without header
        """
        lines = text.split("\n", 1)
        return lines[1].strip() if len(lines) > 1 else lines[0].strip()



