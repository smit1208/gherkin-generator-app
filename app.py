import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os
import datetime

# Load the fine-tuned model and tokenizer
def load_model():
    model_path = "Smit1208/gherkin-generator"  # Your model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

def validate_gherkin(gherkin_text):
    """
    Validates the generated Gherkin text for proper syntax and structure.
    Returns (is_valid, message)
    """
    # Check if text is empty
    if not gherkin_text or len(gherkin_text.strip()) < 10:
        return False, "Generated text is too short or empty."

    # Check for Feature keyword
    if not re.search(r'Feature:', gherkin_text, re.IGNORECASE):
        return False, "Missing 'Feature:' keyword."

    # Check for Scenario keyword
    if not re.search(r'Scenario:', gherkin_text, re.IGNORECASE):
        return False, "Missing 'Scenario:' keyword."

    # Check for Given, When, Then steps
    has_given = re.search(r'\bGiven\b', gherkin_text, re.IGNORECASE)
    has_when = re.search(r'\bWhen\b', gherkin_text, re.IGNORECASE)
    has_then = re.search(r'\bThen\b', gherkin_text, re.IGNORECASE)

    if not (has_given and has_when and has_then):
        missing = []
        if not has_given: missing.append("Given")
        if not has_when: missing.append("When")
        if not has_then: missing.append("Then")
        return False, f"Missing essential steps: {', '.join(missing)}"

    # Check for proper step order (Given should come before When, When before Then)
    given_pos = gherkin_text.lower().find('given')
    when_pos = gherkin_text.lower().find('when')
    then_pos = gherkin_text.lower().find('then')

    if not (given_pos < when_pos < then_pos):
        return False, "Steps are not in the correct order (Given → When → Then)."

    # All checks passed
    return True, "Gherkin syntax is valid! ✅"

def save_feedback(original_input, model_output, feedback_type, user_correction):
    """
    Save user feedback to a CSV file for later model improvement
    """
    feedback_file = "gherkin_feedback.csv"

    feedback_data = {
        "original_input": original_input,
        "model_output": model_output,
        "feedback_type": feedback_type,
        "user_correction": user_correction,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Create DataFrame from feedback
    df_new = pd.DataFrame([feedback_data])

    # Append to existing file or create new one
    if os.path.exists(feedback_file):
        df_existing = pd.read_csv(feedback_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(feedback_file, index=False)
    else:
        df_new.to_csv(feedback_file, index=False)

    return "Thank you for your feedback! We'll use it to improve the model."

def generate_gherkin(input_text, example_selector, max_length, num_beams, temperature):
    # Use example if selected
    if example_selector != "Select an example...":
        input_text = examples[example_selector]

    # Generate Gherkin
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Validate
    validation_result, validation_message = validate_gherkin(generated_text)

    return generated_text, validation_message, validation_result

# Example inputs
examples = {
    "Login Feature": "Create a feature for user login with valid and invalid credentials",
    "Shopping Cart": "Generate scenarios for adding and removing items from a shopping cart",
    "Search Functionality": "Create test scenarios for searching products with different filters",
    "User Registration": "Generate scenarios for user registration with validation checks",
    "Payment Processing": "Create scenarios for payment processing with different payment methods"
}

# Create Gradio interface
with gr.Blocks(title="Gherkin Scenario Generator") as demo:
    gr.Markdown("# Gherkin Scenario Generator")
    gr.Markdown("Enter a feature description to generate Gherkin scenarios for BDD testing")

    with gr.Row():
        with gr.Column():
            # Input section
            gr.Markdown("### Input")

            example_selector = gr.Dropdown(
                ["Select an example..."] + list(examples.keys()),
                label="Try an example or write your own"
            )

            input_text = gr.Textbox(
                lines=8,
                label="Feature Description",
                placeholder="Describe the feature you want to test..."
            )

            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(100, 1000, 512, label="Maximum Length")
                num_beams = gr.Slider(1, 8, 4, label="Beam Search Size")
                temperature = gr.Slider(0.1, 1.5, 1.0, step=0.1, label="Temperature",
                                       info="Higher values make output more random, lower values more deterministic")

            generate_btn = gr.Button("Generate Gherkin", variant="primary")

        with gr.Column():
            # Output section
            gr.Markdown("### Generated Gherkin")

            output_text = gr.Code(language="gherkin", label="Generated Gherkin")
            validation_msg = gr.Textbox(label="Validation")
            is_valid = gr.Checkbox(label="Valid Gherkin", visible=False)

            # Download button (only shows when output is valid)
            download_btn = gr.Button("Download Gherkin", visible=False)

            # Feedback section
            with gr.Accordion("Provide Feedback", open=False):
                gr.Markdown("### Help us improve the model by providing feedback:")

                feedback_type = gr.Radio(
                    ["Yes, it's good", "Needs minor edits", "Needs major changes"],
                    label="Was this Gherkin scenario useful?"
                )

                user_correction = gr.Textbox(
                    lines=6,
                    label="Please provide a corrected version or suggestions:",
                    visible=False
                )

                submit_feedback_btn = gr.Button("Submit Feedback", visible=False)
                feedback_result = gr.Textbox(label="Feedback Result", visible=False)

    # About section
    with gr.Accordion("About Gherkin Format", open=False):
        gr.Markdown("""
        ### Gherkin Format:
        - **Feature:** [Feature name]
        - **Scenario:** [Scenario name]
        - **Given** [precondition]
        - **When** [action]
        - **Then** [expected result]

        ### Tips:
        - Be specific in your feature description
        - Include user roles, actions, and expected outcomes
        - Try different phrasings if results aren't satisfactory
        """)

    # Event handlers
    generate_btn.click(
        generate_gherkin,
        inputs=[input_text, example_selector, max_length, num_beams, temperature],
        outputs=[output_text, validation_msg, is_valid]
    )

    # Show/hide download button based on validation
    def update_download_visibility(is_valid):
        return gr.Button.update(visible=is_valid)

    is_valid.change(
        update_download_visibility,
        inputs=[is_valid],
        outputs=[download_btn]
    )

    # Show/hide correction textbox based on feedback type
    def update_correction_visibility(feedback_type):
        needs_correction = feedback_type != "Yes, it's good"
        return gr.Textbox.update(visible=needs_correction), gr.Button.update(visible=needs_correction)

    feedback_type.change(
        update_correction_visibility,
        inputs=[feedback_type],
        outputs=[user_correction, submit_feedback_btn]
    )

    # Handle feedback submission
    submit_feedback_btn.click(
        save_feedback,
        inputs=[input_text, output_text, feedback_type, user_correction],
        outputs=[feedback_result]
    ).then(
        lambda: gr.Textbox.update(visible=True),
        None,
        [feedback_result]
    )

    # Handle download button
    def download_gherkin(text):
        return text

    download_btn.click(
        download_gherkin,
        inputs=[output_text],
        outputs=[gr.File(label="Download")]
    )

    # Example selector handler
    def update_input(example_name):
        if example_name != "Select an example...":
            return examples[example_name]
        return ""

    example_selector.change(
        update_input,
        inputs=[example_selector],
        outputs=[input_text]
    )

# Launch the app
demo.launch()