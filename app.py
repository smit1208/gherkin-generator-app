# Import libraries
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os
from pyngrok import ngrok
import nest_asyncio

# Apply nest_asyncio to avoid runtime errors
nest_asyncio.apply()

# Create a file for the Streamlit app
%%writefile gherkin_app.py
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os

# Load the fine-tuned model and tokenizer
@st.cache_resource  # This caches the model to avoid reloading
def load_model():
    model_path = "./gherkin_generator_model"  # Update with your model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

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

def save_feedback(feedback_data):
    """
    Save user feedback to a CSV file for later model improvement
    """
    feedback_file = "gherkin_feedback.csv"

    # Create DataFrame from feedback
    df_new = pd.DataFrame([feedback_data])

    # Append to existing file or create new one
    if os.path.exists(feedback_file):
        df_existing = pd.read_csv(feedback_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(feedback_file, index=False)
    else:
        df_new.to_csv(feedback_file, index=False)

    return True

# Main app
def main():
    st.set_page_config(
        page_title="Gherkin Scenario Generator",
        page_icon="📝",
        layout="wide"
    )

    # Load model
    try:
        model, tokenizer = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False

    # App UI
    st.title("Gherkin Scenario Generator")
    st.write("Enter a feature description to generate Gherkin scenarios for BDD testing")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This tool uses AI to generate Gherkin scenarios from feature descriptions.

        **Gherkin Format:**
        - Feature: [Feature name]
        - Scenario: [Scenario name]
        - Given [precondition]
        - When [action]
        - Then [expected result]
        """)

        st.header("Tips")
        st.write("""
        - Be specific in your feature description
        - Include user roles, actions, and expected outcomes
        - Try different phrasings if results aren't satisfactory
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        # Example inputs for users to try
        examples = {
            "Login Feature": "Create a feature for user login with valid and invalid credentials",
            "Shopping Cart": "Generate scenarios for adding and removing items from a shopping cart",
            "Search Functionality": "Create test scenarios for searching products with different filters",
            "User Registration": "Generate scenarios for user registration with validation checks",
            "Payment Processing": "Create scenarios for payment processing with different payment methods"
        }

        # Add example selector
        selected_example = st.selectbox(
            "Try an example or write your own:",
            ["Select an example..."] + list(examples.keys())
        )

        # Text input area
        if selected_example != "Select an example...":
            user_input = st.text_area("Feature Description:", value=examples[selected_example], height=200)
        else:
            user_input = st.text_area("Feature Description:", height=200,
                                      placeholder="Describe the feature you want to test...")

        # Generation parameters
        st.subheader("Generation Settings")
        with st.expander("Advanced Settings"):
            max_length = st.slider("Maximum Length", 100, 1000, 512)
            num_beams = st.slider("Beam Search Size", 1, 8, 4)
            temperature = st.slider("Temperature", 0.1, 1.5, 1.0, 0.1,
                                   help="Higher values make output more random, lower values more deterministic")

        # Generate button
        generate_button = st.button("Generate Gherkin", type="primary", disabled=not model_loaded)

    # Results area
    with col2:
        st.subheader("Generated Gherkin")

        if generate_button and user_input and model_loaded:
            with st.spinner("Generating Gherkin scenarios..."):
                # Tokenize input
                inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)

                # Generate output
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    early_stopping=True
                )

                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Validate the generated Gherkin
                validation_result, validation_message = validate_gherkin(generated_text)

                # Display results
                st.code(generated_text, language="gherkin")

                # Show validation results with color coding
                if validation_result:
                    st.success(validation_message)

                    # Option to download the generated Gherkin
                    st.download_button(
                        label="Download Gherkin",
                        data=generated_text,
                        file_name="generated_scenario.feature",
                        mime="text/plain"
                    )
                else:
                    st.error(validation_message)
                    st.warning("The generated Gherkin may need manual editing.")

                # Feedback section
                st.subheader("Feedback")
                st.write("Help us improve the model by providing feedback:")

                feedback = st.radio(
                    "Was this Gherkin scenario useful?",
                    ["Yes, it's good", "Needs minor edits", "Needs major changes"]
                )

                if feedback != "Yes, it's good":
                    user_correction = st.text_area("Please provide a corrected version or suggestions:")

                    if st.button("Submit Feedback"):
                        # Save the feedback
                        feedback_data = {
                            "original_input": user_input,
                            "model_output": generated_text,
                            "feedback_type": feedback,
                            "user_correction": user_correction,
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        save_feedback(feedback_data)
                        st.success("Thank you for your feedback! We'll use it to improve the model.")
        else:
            st.info("Enter a feature description and click 'Generate Gherkin' to create scenarios.")
            st.image("https://www.specflow.org/wp-content/uploads/2020/09/Gherkin-1.png",
                    caption="Example Gherkin Format", width=400)

if __name__ == "__main__":
    main()

# Set up ngrok tunnel to expose Streamlit app
# Get your authtoken from https://dashboard.ngrok.com/auth
# !ngrok authtoken YOUR_AUTH_TOKEN  # Uncomment and add your token if needed

# Start Streamlit in background
!streamlit run gherkin_app.py &

# Create and display ngrok tunnel
public_url = ngrok.connect(port=8501)
print(f"Streamlit app URL: {public_url}")