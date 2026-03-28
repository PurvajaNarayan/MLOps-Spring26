from pydantic import BaseModel

class ExperienceEntry(BaseModel):
    company: str
    title: str
    dates: str
    bullets: list[str]

class EducationEntry(BaseModel):
    institution: str
    degree: str
    dates: str
    details: list[str] = []

class ParsedResume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = []
    experience: list[ExperienceEntry] = []
    education: list[EducationEntry] = []
    raw_text: str