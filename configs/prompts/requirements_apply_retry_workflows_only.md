Return ONLY the FINAL_WORKFLOWS_JSON block.

Schema summary:
- workflows: list of {id, name, states, transitions}
- transitions: list of {from, to, trigger}
- No extra keys anywhere.

Format:
FINAL_WORKFLOWS_JSON:
{"workflows":[]}

Rules:
- Use "trigger" (not "guard").
- Do not include extra keys.
- Return corrected JSON only, no markdown.
