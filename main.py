import langextract as lx
import textwrap

# 1. Define extraction prompt
prompt = textwrap.dedent("""
    Extract character names and their emotional states from the text.
    Focus on exact text extraction without paraphrasing.
    """)

# 2. Provide few-shot examples
examples = [
    lx.data.ExampleData(
        text="ROMEO: But soft! What light through yonder window breaks?",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder", "scene": "balcony"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe", "intensity": "mild"}
            )
        ]
    )
]

# 3. Execute extraction
result = lx.extract(
    text_or_documents="Juliet gazed longingly at the stars, her heart aching for Romeo.",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    api_key="your-gemini-api-key",
    extraction_passes=2,
    max_workers=10
)

# 4. Display results
for extraction in result.extractions:
    print(f"Class: {extraction.extraction_class}")
    print(f"Text: {extraction.extraction_text}")
    print(f"Attributes: {extraction.attributes}")
