import re
from typing import Dict, List
import tempfile
from pathlib import Path


def text_parser():
    content = """
JOHN DOE
Email: john@example.com
Phone: (123) 456-7890

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in web development
    """
    temp = tempfile.NamedTemporaryFile(suffix='txt', delete=False)
 
    with open(temp.name, 'w') as f:
        f.write(content)

    section_patterns = {
        'summary': r'\b(?:summary|professional\s+summary|profile|objective)\b',
        'experience': r'\b(?:experience|work\s+experience|employment|work\s+history)\b',
        'education': r'\b(?:education|academic|qualifications)\b',
        'skills': r'\b(?:skills|technical\s+skills|competencies)\b',
        'certifications': r'\b(?:certifications|certificates|accreditations)\b',
        'projects': r'\b(?:projects|personal\s+projects|professional\s+projects)\b',
        'other': r'\b(?:additional|interests|volunteer|language|references|personal\s+milestones)\b',
    }

    # Combine all section patterns into one regex with word boundaries
    combined_pattern = re.compile(
        '|'.join(f'(?P<{name}>{pattern})' for name, pattern in section_patterns.items()),
        re.IGNORECASE
    )

    sections = {'header': []}
    current_section = 'header'

    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        match = combined_pattern.match(line)
        if match:
            current_section = match.lastgroup
            if current_section not in sections:
                if current_section: sections[current_section] = []
            continue  # Skip the section heading line itself if needed
        if current_section: sections[current_section].append(line)

    for section_name, lines in sections.items():
        print(f"Section: {section_name.upper()}")
        print('\n'.join(lines))
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    text_parser()
