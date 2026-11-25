"""Base error class of the Tech Market Researcher; every custom error of the project inherits from it."""


class BaseTechMarketResearcherError(RuntimeError):
    """Base class for every custom error of the Tech Market Researcher."""


class SkillsDebugDumpError(BaseTechMarketResearcherError):
    """Raised when dumping the available-skills section to the debug folder fails."""
