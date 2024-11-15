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
    section_header:str
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
            'summary': r'\b(?:PROFESSIONAL\s+SUMMARY|SUMMARY|PROFILE|OBJECTIVE)\b:?',
            'experience': r'\b(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|WORK\s+HISTORY)\b:?',
            'education': r'\b(?:EDUCATION|ACADEMIC|QUALIFICATIONS)\b:?',
            'skills': r'\b(?:SKILLS|TECHNICAL\s+SKILLS|COMPETENCIES)\b:?',
            'certifications': r'\b(?:CERTIFICATIONS|CERTIFICATES|ACCREDITATIONS)\b:?',
            'projects': r'\b(?:PROJECTS|PERSONAL\s+PROJECTS|PROFESSIONAL\s+PROJECTS)\b:?',
            'other': r'\b(?:ADDITIONAL|INTERESTS|VOLUNTEER|LANGUAGE|REFERENCES)\b:?',
        }
        self.combined_patterns = re.compile(
            '|'.join(f'(?P<{name}>{pattern})' for name,pattern in self.section_patterns.items()),
            re.IGNORECASE
        )

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
            # logger.info(f"🔍 structured_content: {structured_content}")

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
        """
        Parse plain text documents
        
        Args:
            file_path: Path to the text file

        Returns:
            Content of the text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeError:
            # Try different encodings if UTF-8 fails
            encodings = ['latin-1', 'iso-8859','cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeError:
                    continue
            raise ValueError('❌ Unable to decode the text file with supported encodings')

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
        # logger.info(f"🔍 cleaned_text: {cleaned_text}")
        # Detect sections
        sections = self._detect_sections(cleaned_text)
        # logger.info(f"🔍 sections: {sections}")
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

        for section in sorted_sections:
                        # Store in appropriate section
            if section.name in structured_content:
                structured_content[section.name] = section.content
            else:
                structured_content['other'] += f"\n{section.content}" if structured_content['other'] else section.content



        # logger.info(f"🔍 structured_content: {structured_content}")
        return structured_content

    def print_sections(self, sections:List[Section]):
        for section in sections:
            print(section)

    def _detect_sections(self, text:str) -> List[Section]:
        """
        Detect document sections using regex patterns as NLP

        Args:
            text: Cleaned document text

        Returns:
            List of detected sections with their positions
        """
        
        # Find all matches of section headings
        matches = list(self.combined_patterns.finditer(text))

        sections = []

        # Handling the header section with personal contact info (content before first matched section)
        if matches and matches[0].start() > 0:
            header_content = text[:matches[0].start()]
            header = Section(
                name='header',
                section_header = '',
                content = header_content.strip(),
                start_idx = 0,
                end_idx = matches[0].start()
            )
            sections.append(header)
        elif not matches:
            # No matches found; entire content is header
            header = Section(
                name = 'header',
                section_header = '',
                content = text.strip(),
                start_idx = 0,
                end_idx = len(text)
            )
            sections.append(header)



        for idx, match in enumerate(matches):
            name = match.lastgroup
            section_header = match.group()
            if self._verify_section_header(section_header.strip()):
                start_idx = match.end() # Content starts after the section heading

                # Determine the end index
                if idx + 1 < len(matches):
                    end_idx = matches[idx + 1].start()
                else:
                    end_idx = len(text)

                section_content = text[start_idx:end_idx]
                sections.append(
                    Section(
                        name = name if name else '',
                        section_header = section_header.strip(),
                        content = section_content,
                        start_idx = start_idx,
                        end_idx = end_idx
                    )
                )
        
        return sections


    def _validate_parsed_content(self, content:Dict[str, str]) -> None:
        """
        Validate parsed content structure and content

        Args:
            content: Structured content dictionary to validate

        Raises:
            ValueError: If content structure is invalid
        """
        # Check found sections
        found_sections = {section_name for section_name, content in content.items() if content}
        if found_sections:
            logger.info(f"ℹ️  Found sections: {found_sections}")
        # Check for required sections
        required_sections = {'header','summary','experience','education','skills'}
        missing_sections = required_sections - set(content.keys())
        if missing_sections:
            logger.warning(f"⚠️  Missing required sections: {missing_sections}")

        # Check for empty sections
        empty_sections = [section for section, text in content.items() if not text.strip()]
        if empty_sections:
            logger.warning(f" 🔍 Empty sections detected: {empty_sections}")

        # Basic content validation
        for section, text in content.items():
            if not isinstance(text, str):
                raise ValueError(f"Invalid content type in section '{section}': expected str, got {type(text)}")

            # Check for potential parseing errors (e.g, garbage characters)
            if re.search(r'[^\x00-\x7F]+', text):
                logger.warning(f"⚠️  Non-ASCII characters detected in section '{section}'")


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





