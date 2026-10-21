# Name research (2026-10-21)

> Research from 2026-10-21, done to choose this project's public name; registry and repository
> figures were current then.

Method: PyPI JSON API + npm registry HTTP status (404 = free), GitHub search API (name match, by stars; rate-limited, so some rows are web-search only), official MCP registry search, web search. Not checked: Hugging Face (no result captured), mcp.so/Smithery direct, USPTO live, domains.

| name | GitHub collisions | PyPI | npm | other collisions | SEO | verdict |
|---|---|---|---|---|---|---|
| video-watch | No repo with this exact name in top hits. Neighbors: Newuxtreme/watch-video-skill (72 stars, "Claude skill that teaches AI to watch videos"), MohamedBakoush/WatchVideoByLink (157, unrelated) | free | free | MCP registry: none. USPTO "VIDEO WATCH" (1990, rental) and "VIDEO WATCHDOG" both cancelled. No product found | Very generic, matches "watch video" searches; hard to rank alone, needs keywords (claude, agent) in description | OK, safe but crowded neighborhood |
| videowatch | sathvik3115/videowatch (4), lordbooga/videowatch (1), KJCondron/VideoWatcher (5) - tiny, unrelated | free | free | none found | Same as above, one token | OK |
| watch-video | Newuxtreme/watch-video-skill (72, same niche), TomGranot/watch-video (7) | free | free | same niche neighbor | very generic | Avoid (too close to watch-video-skill) |
| agent-watch / agentwatch | many: AIAnytime/agent-watch, different-ai/agent-watch, cyberark/agentwatch (126), dongxutang918-afk/agentwatch (111) | taken (agent-watch) | taken | agent observability meaning | wrong semantics | Avoid |
| vidsense | Rashipotey/VidSense (8) etc., small | free | free | VideoSenseAI, Vidsight, VizioSense (different) | ok | Possible, weak |
| watchkit | not queried (Apple WatchKit is a major framework) | free | free | Apple WatchKit | collides with Apple docs | Avoid |
| eyes-and-ears | tiny unrelated (EyesAndEars 4) | free | free | none in niche | descriptive, poor keyword for search | Good tagline, not a package name |
| video-eyes | SpinachDigital/video-eyes (1) | free | free | none seen | fair | OK |
| skim | n/a | taken | taken | many | n/a | Avoid |
| vidlens | thatsrajan/vidlens-mcp (34), francis20060728-jpg/vidlens (23, "text-only AI agents see images and video"), lordfine/vidlens (4, agent video evidence) | free | free | MCP registry: io.github.thatsrajan/vidlens-mcp | same niche | Avoid |
| seevid | none | free | free | SeeVid and SeeVido AI (AI video generators) | brand collision in video-AI | Avoid |
| tubewatch | none | free | free | none seen | YouTube-only feel | OK but too narrow |
| vidwatch | luquitared/vidwatch ("Give agents eyes for video"), louisjournals/vidwatch (agent skill), tickernelz/vidwatch (MCP, meeting recordings) | free | free | none else | 3 same-niche repos | Avoid |
| watchvid / watchvideo | charlesdove977/watchvideo (23, Claude Code video skill), hannahinthere/watchvideo | watchvid free | free | same niche | | Avoid |
| videolens / watchlens / seehear / vidhear / vidgaze / vidsonar | none seen (GitHub only partially checked) | free (video-lens taken) | free | none found in web search | videolens generic-ish | Candidates to vet |

Other niche projects seen (avoid similarity): oxbshw/watch-skill (~235 stars, PyPI watch-skill), bradautomates/claude-video (~7k), taoufik123-collab/claude-watch, HUANGCHIHHUNGLeo/claude-real-video, mnvsk97/eyeroll, vidistill (MCP/CLI), VideoDB.

Sources: pypi.org/pypi/<name>/json, registry.npmjs.org/<name>, api.github.com/search/repositories, registry.modelcontextprotocol.io, trademark.justia.com (VIDEO WATCH 74020771, VIDEO WATCHDOG 74664271), pypi.org/project/watch-skill, glama.ai/mcp/servers/thatsrajan/vidlens-mcp, creati.ai/ai-tools/seevid, web searches.
