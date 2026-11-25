"""Dump the available-skills section the agent offers to the model, to the debug folder."""

import logging
import shutil
from pathlib import Path
from types import MethodType
from typing import Any, cast

from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from tech_market_researcher.core.constants import (
    BASE_DEBUG_DIR_PATH,
    DEBUG_AGENT_FOLDER,
    DEBUG_SKILLS_FOLDER,
    SKILLS_FILE_NAME,
)
from tech_market_researcher.core.errors import SkillsDebugDumpError

logger = logging.getLogger(__name__)

# Graph node langchain compiles the skills middleware's before_agent hook into.
SKILLS_MIDDLEWARE_NODE_NAME = "SkillsMiddleware.before_agent"


def _agent_skills_middleware(agent: CompiledStateGraph[Any, Any]) -> SkillsMiddleware | None:
    """Return the SkillsMiddleware the agent was built with, or None if the agent has no skills.

    The middleware's hooks are compiled into the graph as nodes that bind the middleware's methods,
    so the instance can be recovered from the compiled agent and is always the one the model sees.
    """
    try:
        node = agent.nodes[SKILLS_MIDDLEWARE_NODE_NAME]
    except KeyError:
        logger.debug("Agent has no %s node; it is not built with skills", SKILLS_MIDDLEWARE_NODE_NAME)
        return None
    func = getattr(node.bound, "func", None)
    if not (isinstance(func, MethodType) and isinstance(func.__self__, SkillsMiddleware)):
        msg = f"node '{SKILLS_MIDDLEWARE_NODE_NAME}' does not bind a SkillsMiddleware; cannot dump the skills"
        raise SkillsDebugDumpError(msg)
    return cast(SkillsMiddleware, func.__self__)


def dump_agent_skills(agent: CompiledStateGraph[Any, Any]) -> Path:
    """Write the skills section injected into the agent's system prompt to the debug folder.

    The SkillsMiddleware is recovered from the compiled agent, and the skills are loaded through
    its own loader, so the rendered section matches exactly what the model sees: name, description
    and path per skill, per the progressive-disclosure pattern. The dump folder is emptied first,
    so files from an earlier dump do not linger.
    """
    out_dir = BASE_DEBUG_DIR_PATH / DEBUG_AGENT_FOLDER / DEBUG_SKILLS_FOLDER
    if out_dir.exists():
        shutil.rmtree(out_dir)
    middleware = _agent_skills_middleware(agent)
    if middleware is None:
        logger.info("Agent is not built with skills; nothing to dump in %s", out_dir)
        return out_dir
    # before_agent only loads the skills from the backend; it ignores runtime and config, and an
    # empty state means the skills are not yet loaded, so it always returns an update.
    update = middleware.before_agent(
        {"messages": []},
        cast(Runtime, None),
        cast(RunnableConfig, {}),
    )
    if update is None:
        msg = "the skills middleware returned no skills update; cannot dump the skills"
        raise SkillsDebugDumpError(msg)
    if middleware.system_prompt_template is None:
        msg = "the skills middleware has no system prompt template; cannot dump the skills"
        raise SkillsDebugDumpError(msg)
    skills_locations = middleware._format_skills_locations()  # pylint: disable=protected-access
    # pylint: disable-next=protected-access
    skills_load_warnings = middleware._format_skills_load_warnings(update["skills_load_errors"])
    skills_list = middleware._format_skills_list(update["skills_metadata"])  # pylint: disable=protected-access
    skills_section = middleware.system_prompt_template.format(
        skills_locations=skills_locations,
        skills_load_warnings=skills_load_warnings,
        skills_list=skills_list,
    )
    out_path = out_dir / SKILLS_FILE_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(skills_section + "\n", encoding="utf-8")
    logger.info("Dumped the available-skills section (%d skill(s)) to %s", len(update["skills_metadata"]), out_path)
    return out_path
