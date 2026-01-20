Return ONLY the FINAL_DOMAIN_MODEL_JSON block.

Schema summary:
- entities: list of {name, description, attributes}
- attributes: list of {name, type, description}
- relationships: list of {from, to, type, description}
- No extra keys anywhere.

Format:
FINAL_DOMAIN_MODEL_JSON:
{"entities":[],"relationships":[]}

Rules:
- Ensure every relationship has a description.
- Return corrected JSON only, no markdown.
