import re
import pdfplumber
from io import BytesIO
from app.models import ParsedResume, ExperienceEntry, EducationEntry

# Section headers commonly found in resumes
SECTION_PATTERNS = {
    "experience": re.compile(r"^(experience|work experience|professional experience|employment)", re.I),
    "education": re.compile(r"^(education|academic)", re.I),
    "skills": re.compile(r"^(skills|technical skills|technologies|tools)", re.I),
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{10,15}")


def extract_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


def split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"header": []}
    current = "header"

    for line in lines:
        matched = False
        for section, pattern in SECTION_PATTERNS.items():
            if pattern.match(line.strip()):
                current = section
                sections.setdefault(current, [])
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_contact(header: str) -> dict:
    lines = [l.strip() for l in header.split("\n") if l.strip()]
    name = lines[0] if lines else None
    email_match = EMAIL_RE.search(header)
    phone_match = PHONE_RE.search(header)
    return {
        "name": name,
        "email": email_match.group() if email_match else None,
        "phone": phone_match.group().strip() if phone_match else None,
    }


def parse_skills(skills_text: str) -> list[str]:
    # Handle comma-separated, pipe-separated, or bullet-separated skills
    skills_text = re.sub(r"[•·|]", ",", skills_text)
    raw = [s.strip() for s in skills_text.split(",")]
    # Also split on newlines for list-style resumes
    expanded = []
    for s in raw:
        expanded.extend(s.split("\n"))
    return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 1]


def parse_experience(exp_text: str) -> list[ExperienceEntry]:
    """V1: Split on date-like patterns. Intentionally simple — 
    will upgrade to LLM-based extraction in V2."""
    entries = []
    # Split on lines that look like they contain date ranges
    date_re = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*[\s,]+\d{4})"
        r".*?(?:–|—|-|to|present)",
        re.I,
    )
    blocks = re.split(r"\n(?=\S)", exp_text)

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        date_match = date_re.search(block)
        title = lines[0] if lines else ""
        company = lines[1] if len(lines) > 1 else ""
        dates = date_match.group() if date_match else ""
        bullets = [l.lstrip("•·-– ") for l in lines[2:] if l.strip()]

        if title or company:
            entries.append(ExperienceEntry(
                company=company, title=title, dates=dates, bullets=bullets
            ))
    return entries


def parse_education(edu_text: str) -> list[EducationEntry]:
    blocks = re.split(r"\n(?=\S)", edu_text)
    entries = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        entries.append(EducationEntry(
            institution=lines[0],
            degree=lines[1] if len(lines) > 1 else "",
            dates=lines[2] if len(lines) > 2 else "",
            details=[l for l in lines[3:] if l],
        ))
    return entries


def parse_resume(pdf_bytes: bytes) -> ParsedResume:
    raw = extract_text(pdf_bytes)
    sections = split_sections(raw)

    contact = parse_contact(sections.get("header", ""))
    skills = parse_skills(sections.get("skills", ""))
    experience = parse_experience(sections.get("experience", ""))
    education = parse_education(sections.get("education", ""))

    return ParsedResume(
        name=contact["name"],
        email=contact["email"],
        phone=contact["phone"],
        skills=skills,
        experience=experience,
        education=education,
        raw_text=raw,
    )