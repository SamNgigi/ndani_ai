import pytest
import tempfile
from pathlib import Path

from app.components.document_processor import DocumentParser, Section


class TestDocumentParser:
    @pytest.fixture
    def parser(self):
        return DocumentParser()

    @pytest.fixture
    def sample_resume_txt(self):
        content = """
JOHN DOE
Email: john@example.com
Phone: (123) 456-7890

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in web development

WORK EXPERIENCE
Senior Developer - Tech Corp
2018-Present
- Led development of cloud-based applications
- Managed team of 5 developers

EDUCATION
BS Computer Science
University of Technology, 2015

SKILLS
- Python, JavaScript, React
- Cloud platforms (AWS, AZURE)
- Agile methodologies
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
        yield Path(f.name)
        Path(f.name).unlink()


    # def test_parse_txt(self, parser:DocumentParser, sample_resume_txt:Path):
    #     """Test parsing of text file"""
    #     result = parser.parse(sample_resume_txt)
    #     print(result)
    
    def test_detect_section(self, parser):
        """Test setion detection"""
        text = """
SUMMARY
Test summary

EXPERIENCE
Test Experience

EDUCATION
Test Education
"""
        sections = parser._detect_sections(text)

        # Check if all sections are detected
        section_names = [s.name for s in sections]
        assert 'summary' in section_names
        assert 'experience' in section_names
        assert 'education' in section_names

        assert sections[0].start_idx < sections[1].start_idx
        assert sections[1].start_idx < sections[2].start_idx

    def test_verify_section_header(self, parser):
        """Test section header verification"""
        # Should be identified as headers
        assert parser._verify_section_header("PROFESSIONAL SUMMARY")
        assert parser._verify_section_header("Work Experience")
        assert parser._verify_section_header("Education:")

        # Should not be identified as headers
        assert not parser._verify_section_header("Developed full stack application")
        assert not parser._verify_section_header("This is a very long line containiing too many works to be in a header")
        assert not parser._verify_section_header("2018-2020: Software Developer")

    def test_clean_text(self, parser: DocumentParser):
        """Test text cleaning"""
        dirty_text = """
        Test • point 1
        


        Test ● point 2
        

        Test ◘ point 3
        



        Test ▫ point 4

        Page 1
        """
        cleaned = parser._clean_text(dirty_text)
        # Check cleaning results
        assert '•' not in cleaned
        assert '●' not in cleaned
        assert '◘' not in cleaned
        assert '▫' not in cleaned
        assert 'Page 1' not in cleaned
        assert cleaned.count('\n') < dirty_text.count('\n')
