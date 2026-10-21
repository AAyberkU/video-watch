# Harness packaging research (2026-10-21)

> Research from 2026-10-21; agent paths, plugin formats and tool support were current then and
> change often. Check the linked official docs before relying on a detail.

Method: web fetch of official docs plus web search. "UNVERIFIED" means not confirmed from an official page.

## Comparison

| | Claude Code | Codex (CLI+IDE) | OpenCode | Kilo (VS Code ext + CLI) |
|---|---|---|---|---|
| SKILL.md support | Yes | Yes | Yes | Yes |
| Project path | `.claude/skills/<name>/SKILL.md` (+ nested subdirs) | `.agents/skills/` in CWD, parents, repo root | `.opencode/skills/`, `.claude/skills/`, `.agents/skills/` (walks up to git root) | `.kilo/skills/`; also `.agents/skills/`, `.claude/skills/` (latter only if Claude compat enabled) |
| User path | `~/.claude/skills/` (Win: `%USERPROFILE%\.claude\skills`) | `~/.agents/skills/`; admin `/etc/codex/skills` | `~/.config/opencode/skills/`, `~/.claude/skills/`, `~/.agents/skills/` | `~/.kilo/skills/` (Win `\Users\<u>\.kilo\skills\`); `~/.kilocode/skills` appears to be the legacy path |
| Frontmatter | name/description plus extras (when_to_use, allowed-tools, disable-model-invocation, context...). description+when_to_use truncated at 1,536 chars in listing; body under 500 lines advised | name+description required; optional `agents/openai.yaml` (policy.allow_implicit_invocation, dependencies.tools). Skill list budget about 2% of context (8k chars fallback) | name+description required; license/compatibility/metadata optional, unknown ignored. name = lowercase-hyphen and must equal dir name; description 1-1024 chars | name+description; name must match folder; optional scripts/ references/ assets/ |
| Loading | Progressive: description every turn, body on invoke | Progressive: name+desc+path, body when selected | On demand via native skill tool | Metadata at start, body on match; each skill also a slash command; shell placeholders need approval |
| Plugin/marketplace | Yes: plugin = dir with `.claude-plugin/plugin.json` + `skills/`; marketplace = git repo with `.claude-plugin/marketplace.json`; `/plugin marketplace add owner/repo`, `/plugin install name@market` | Yes: plugin = root `plugin.json` (or `.codex-plugin/plugin.json`) + `skills/<n>/SKILL.md`; marketplace `.agents/plugins/marketplace.json`; `codex plugin marketplace add owner/repo`; public directory via submission portal | Plugins are JS/TS (npm `plugin` array in opencode.json, or `.opencode/plugins/`); docs fetched do not show skill bundling; no marketplace, only a community list. Skills can be added via `skills` array (dirs / HTTP index) in opencode.json | Kilo-Org/kilo-marketplace GitHub repo: PR a skill folder (SKILL.md) or an MCP def in `mcps/` with optional companion `skills`; also remote skill URLs via `index.json`; `skills` paths in kilo.jsonc |
| Reads image files | Yes, Read tool returns images | Yes: `view_image` tool; `-i/--image` flag (IDE ext UNVERIFIED) | Read tool detects PNG/JPEG/GIF/WebP, max 20 MiB, auto-resize; model must support vision; open bug anomalyco/opencode#20802 with custom OpenAI-compatible providers | UNVERIFIED in docs (read tool listed, image support not stated); depends on vision model |
| Shell | Bash/PowerShell tool, permission prompts; skill `allowed-tools: Bash(...)` pre-approves | Yes, sandbox + approval policy | Yes (`shell`/bash tool), `permission` allow/ask/deny | Yes, approval/auto-approve settings |
| MCP | Yes | Yes; image content supported since 2025, model sees it (PR #5600) | Yes; image-result handling UNVERIFIED | Yes (ask agent needs approval per call); image handling UNVERIFIED |

## Cross-harness conventions
- `.agents/skills/` is the de-facto shared project dir: Codex native; OpenCode and Kilo also scan it; Claude Code does NOT (uses `.claude/skills/`). Universal global differs: `~/.config/agents/skills/` (vercel skills CLI "universal") vs `~/.agents/skills/` (Codex).
- vercel-labs/skills (`npx skills add owner/repo`, Node >=22.20): detects installed agents and installs the SKILL.md folder into each agent's dir (project by default, `-g` global, `-a claude-code -a codex -a opencode -a kilo`, `--skill`, `--copy` to avoid symlinks). Agent table: claude-code `.claude/skills` / `~/.claude/skills`; codex `.agents/skills` / `~/.agents/skills`; opencode `.agents/skills` / `~/.config/opencode/skills`; kilo `.agents/skills` / `~/.kilo/skills`. Claims 50-75+ agents (count varies by version). Also find/list/update/remove/init. Symlink-vs-copy default not confirmed from official page.
- claude-video (bradautomates): README says `npx skills add bradautomates/claude-video -g` (this is how it reaches "50+" hosts), plus Claude Code plugin route `/plugin marketplace add bradautomates/claude-video` then `/plugin install watch@claude-video`, plus a release zip for claude.ai upload. (From third-party instagit.com summaries, not the repo itself.)
- Codex `$skill-installer` handles curated skills only; docs recommend plugins for distributing your own.

## MCP alternative
All four support MCP. Image display from MCP results is documented only for Codex; unverified for OpenCode/Kilo. An MCP server needs a running process and per-harness config in four formats, and still benefits from a skill for "how to read output". CLI + SKILL.md relies on image file reading, which works in all four given a vision model. MCP is a possible optional extra, not the primary package.

## Recommendation
1. Repo layout: `skills/<name>/SKILL.md` (+ scripts/references). Portable frontmatter: name = dir name, lowercase-hyphen, description under 1024 chars, key trigger first. Do not rely on Claude-only fields.
2. Primary install: `npx skills add owner/repo -g` (covers all four). Document manual copy fallbacks: `.agents/skills/` (Codex/OpenCode/Kilo) and `.claude/skills/` (Claude Code).
3. SKILL.md runs the CLI via `uv run --project <path relative to skill dir>` or `uvx --from git+https://...`; CLI writes frames to a known dir and prints absolute paths; instruct "view image files with your read/view tool; if the model has no vision, use text output only".
4. Claude Code plugin: worth adding (one `.claude-plugin/plugin.json` + `.claude-plugin/marketplace.json` pointing at the same `skills/` dir; gives `/plugin install` and versioned updates; skills become namespaced). Codex plugin (`plugin.json` + `.agents/plugins/marketplace.json`) is similarly cheap, optional. Skip OpenCode plugin. Kilo: optional PR to Kilo-Org/kilo-marketplace later.
5. Test image reading separately in OpenCode, Kilo (IDE and CLI) and Codex IDE before claiming support.

## Sources
- Claude skills: https://code.claude.com/docs/en/skills
- Claude plugins: https://code.claude.com/docs/en/plugins ; https://code.claude.com/docs/en/plugins/create-marketplace
- Codex skills: https://developers.openai.com/codex/skills (redirects to https://learn.chatgpt.com/docs/build-skills.md)
- Codex plugins: https://developers.openai.com/plugins/build/plugins
- OpenCode: https://opencode.ai/docs/skills ; https://opencode.ai/docs/plugins/ ; https://opencode.ai/docs/tools/ ; https://v2.opencode.ai/attachments
- Kilo: https://kilo.ai/docs/customize/skills ; https://github.com/Kilo-Org/kilo-marketplace ; https://github.com/Kilo-Org/skills
- vercel skills CLI: https://github.com/vercel-labs/skills (docs/agents.md)
- claude-video: https://github.com/bradautomates/claude-video (via instagit.com summaries)
- Codex images: https://codex.danielvaughan.com/2026/03/28/codex-cli-image-workflows/
