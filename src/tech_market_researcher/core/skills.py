"""Skill bundle validation and materialization for the deep agent."""

import logging
import re
from pathlib import Path

import yaml
from deepagents.backends import BackendProtocol

from tech_market_researcher.core.agent_backend import delete_backend_path
from tech_market_researcher.core.configuration import SkillBundleConfig
from tech_market_researcher.core.constants import SKILL_BUNDLES_DIR, SKILLS_BACKEND_DIR

logger = logging.getLogger(__name__)

_SKILL_FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _skill_frontmatter(skill_md: Path) -> dict[str, str]:
    """Parse the SKILL.md YAML frontmatter into a dict of string fields.

    Fails if the frontmatter is missing or if the required 'name' or 'description' fields are
    absent or empty.
    """
    match = _SKILL_FRONTMATTER_PATTERN.match(skill_md.read_text(encoding="utf-8"))
    if match is None:
        msg = f"{skill_md}: no YAML frontmatter found"
        raise ValueError(msg)
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        msg = f"{skill_md}: frontmatter is not a mapping"
        raise ValueError(msg)
    for required_field in ("name", "description"):
        if not str(frontmatter.get(required_field) or "").strip():
            msg = f"{skill_md}: missing '{required_field}' in frontmatter"
            raise ValueError(msg)
    return {str(key): str(value) for key, value in frontmatter.items()}


def _bundle_source_dir(bundle: SkillBundleConfig) -> Path:
    """Return the on-disk directory of the bundle's version."""
    return SKILL_BUNDLES_DIR / bundle.bundle_name / bundle.version


def _check_skill_name(name: str, directory_name: str, skill_md: Path) -> None:
    """Check a skill name against the Agent Skills spec and its skill directory name.

    The deep agent only logs a warning and skips a skill with an invalid name, so the name is
    checked here to fail at startup instead of silently dropping the skill.
    """
    if name != directory_name:
        msg = f"{skill_md}: frontmatter name '{name}' must match directory name '{directory_name}'"
        raise ValueError(msg)
    if (
        len(name) > 64
        or name.startswith("-")
        or name.endswith("-")
        or "--" in name
        or not all(c == "-" or (c.isalpha() and c.islower()) or c.isdigit() for c in name)
    ):
        msg = (
            f"{skill_md}: skill name '{name}' must be 1-64 characters of lowercase alphanumerics"
            " and single hyphens, and must not start or end with a hyphen"
        )
        raise ValueError(msg)


def validate_skill_bundles(skill_bundles: list[SkillBundleConfig]) -> None:
    """Check the enabled bundles: their version directories exist, and their skill names are
    valid and unique.

    Every SKILL.md must carry a valid frontmatter, and its name must be valid per the Agent
    Skills spec and match the skill's directory name. Skill names are also checked for
    duplicates because the deep agent would otherwise silently override them (last source wins).
    """
    skill_name_owners: dict[str, str] = {}
    for bundle in skill_bundles:
        if not bundle.enabled:
            continue
        source_dir = _bundle_source_dir(bundle)
        if not source_dir.is_dir():
            msg = f"skill bundle '{bundle.bundle_name}' version '{bundle.version}' not found at {source_dir}"
            raise ValueError(msg)
        for skill_md in sorted(source_dir.glob("*/SKILL.md")):
            name = _skill_frontmatter(skill_md)["name"]
            _check_skill_name(name, skill_md.parent.name, skill_md)
            owner = f"{bundle.bundle_name}@{bundle.version} ({skill_md.parent.name})"
            if (prev_owner := skill_name_owners.get(name)) is not None:
                msg = f"skill name '{name}' is defined by both {prev_owner} and {owner}"
                raise ValueError(msg)
            skill_name_owners[name] = owner


def materialize_skill_bundles(backend: BackendProtocol, skill_bundles: list[SkillBundleConfig]) -> list[str]:
    """Upload each enabled skill bundle to the agent's backend, under skills/<skill name>.

    The deep agent reads skills only from the backend, so the bundles are written there through
    the backend's upload API, which works for any backend (not just an on-disk one). The entire
    skills directory is reset first, so files removed from the repo (or added manually in the
    backend) do not linger. Skills are flattened into the backend without their bundle name or
    version, relying on the skill name uniqueness check done during validation.
    Returns the skill source directory the deep agent should read skills from; it lists the
    skill subdirectories of every source path it is given, so the whole flattened directory is
    passed as a single source.
    """
    validate_skill_bundles(skill_bundles)
    delete_backend_path(backend, f"/{SKILLS_BACKEND_DIR}")
    any_enabled = False
    for bundle in skill_bundles:
        if not bundle.enabled:
            logger.info("Skipping skill bundle %s (disabled)", bundle.bundle_name)
            continue
        any_enabled = True
        source_dir = _bundle_source_dir(bundle)
        files = [
            (f"{SKILLS_BACKEND_DIR}/{path.relative_to(source_dir).as_posix()}", path.read_bytes())
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        ]
        responses = backend.upload_files(files)
        failed = [response.path for response in responses if response.error]
        if failed:
            msg = f"failed to upload skill files for bundle '{bundle.bundle_name}': {failed}"
            raise ValueError(msg)
        logger.info("Materialized skill bundle %s@%s from %s", bundle.bundle_name, bundle.version, source_dir)
    if not any_enabled:
        return []
    return [SKILLS_BACKEND_DIR]
