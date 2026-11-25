# Skill Bundles

Agent skills follow the [Agent Skills specification](https://agentskills.io/specification): each
skill is a directory containing a `SKILL.md` (YAML frontmatter with `name` and `description`,
followed by the instructions), optionally with supporting files under `scripts/`, `references/`,
and `assets/`. A skill name must be 1-64 characters of lowercase letters, digits and single
hyphens (no leading, trailing or consecutive hyphens), and match its directory name.

Layout:

```
skill-bundles/
  <bundle name>/
    <bundle version>/
      <skill name>/
        SKILL.md
```

A bundle can contain a single skill; the bundle+version is the unit the agent config selects.

The enabled bundles are declared in the agent config as a list of items with `bundle_name` and
`version`, each of which can be disabled with an `enabled` flag.
