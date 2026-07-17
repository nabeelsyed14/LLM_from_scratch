import os
import sys
import torch
import tiktoken
import gradio as gr

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import GPTModel
from src.config import GPT_CONFIG_124M

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and checkpoints
model = GPTModel(GPT_CONFIG_124M)
checkpoint_path = os.path.join("checkpoints", "model_instruction_finetuned.pth")

if os.path.exists(checkpoint_path):
    # Load state_dict directly
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded SFT model checkpoint from {checkpoint_path}")
else:
    print(f"CRITICAL ERROR: Checkpoint not found at {checkpoint_path}! Please check file location.")

model.to(device)
model.eval()

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def format_input(instruction, input_text):
    """Format input according to Alpaca SFT template."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    input_text = f"\n\n### Input:\n{input_text}" if input_text else ""
    return instruction_text + input_text

def stream_generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generates text token by token, yielding the sequence at each step."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-K filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Sampling or greedy selection
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop if EOS is hit
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        yield idx

def run_generation(instruction, context_input, temperature, top_k, max_new_tokens):
    """Wrapper that formats prompt and calls generator, yielding clean output text."""
    if not instruction.strip():
        yield "Error: Instruction cannot be empty."
        return

    prompt = format_input(instruction.strip(), context_input.strip())
    prompt_with_response = prompt + "\n\n### Response:\n"
    
    token_ids = tokenizer.encode(prompt_with_response)
    token_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)
    
    context_size = GPT_CONFIG_124M["context_length"]
    
    for idx_tensor in stream_generate(
        model, 
        token_tensor, 
        max_new_tokens=int(max_new_tokens), 
        context_size=context_size, 
        temperature=temperature, 
        top_k=top_k, 
        eos_id=50256
    ):
        generated_text = tokenizer.decode(idx_tensor[0].tolist())
        # Stream only the newly generated response text
        response_text = generated_text[len(prompt_with_response):].strip()
        yield response_text

def chat_stream(message, history, temperature, top_k, max_new_tokens):
    """Streaming function for standard Gradio ChatInterface."""
    # Gradio ChatInterface yields the latest response
    for response in run_generation(message, "", temperature, top_k, max_new_tokens):
        yield response

# CSS code for a premium, custom styled look
custom_css = """
footer {display: none !important;}
.gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    padding: 2.5rem;
    border-radius: 1.2rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
}
.header-box h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.025em;
}
.header-box p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-top: 0.5rem;
    font-weight: 300;
}
.setting-card {
    border-radius: 1rem !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    padding: 1.5rem !important;
    background: #f8fafc !important;
}
.dark .setting-card {
    background: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
)

with gr.Blocks(title="GPT-2 SFT Playground") as demo:
    gr.HTML("""
        <div class="header-box">
            <h1>🧠 GPT-2 SFT Playground</h1>
            <p>Interact with your custom instruction-finetuned GPT-2 124M model in real-time</p>
        </div>
    """)
    
    with gr.Row():
        # Left column: Controls / Model specifications
        with gr.Column(scale=1, elem_classes="setting-card"):
            gr.Markdown("### ⚙️ Generation Settings")
            temp_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.5, 
                value=0.0, 
                step=0.1, 
                label="Temperature",
                info="0.0 = deterministic (greedy), >0.0 = creative/random."
            )
            top_k_slider = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=50, 
                step=1, 
                label="Top-K Filter",
                info="Limits predictions to the top K most likely tokens."
            )
            max_tokens_slider = gr.Slider(
                minimum=10, 
                maximum=512, 
                value=256, 
                step=10, 
                label="Max New Tokens",
                info="Max tokens the model will generate."
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Model Specifications")
            device_text = f"🟢 **Device:** `{device.type.upper()}`"
            gr.Markdown(f"""
            {device_text}
            - **Base Model:** GPT-2 (124M parameters)
            - **Context Length:** 1024 tokens
            - **Dataset Size:** 1,100 Q&A pairs (Alpaca SFT)
            - **Loss Masking:** Enabled (penalizes only output responses)
            """)
            
        # Right column: Tabs for different interaction modes
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("📋 Alpaca SFT Mode"):
                    gr.Markdown("Test the model using its exact training format (Instruction + optional Input context).")
                    
                    instruction_input = gr.Textbox(
                        label="Instruction",
                        placeholder="e.g., Rewrite the sentence using a simile.",
                        lines=3
                    )
                    context_input = gr.Textbox(
                        label="Input Context (Optional)",
                        placeholder="e.g., The car is very fast.",
                        lines=2
                    )
                    
                    submit_button = gr.Button("Generate Response", variant="primary")
                    
                    response_output = gr.Textbox(
                        label="Model Response",
                        placeholder="The model response will stream here...",
                        lines=6,
                        interactive=False
                    )
                    
                    submit_button.click(
                        fn=run_generation,
                        inputs=[instruction_input, context_input, temp_slider, top_k_slider, max_tokens_slider],
                        outputs=response_output
                    )
                    
                with gr.Tab("💬 Simple Chat Mode"):
                    gr.Markdown("A standard chat interface. Note: Multi-turn history is skipped; each chat input is treated as a clean instruction prompt.")
                    
                    gr.ChatInterface(
                        fn=chat_stream,
                        additional_inputs=[temp_slider, top_k_slider, max_tokens_slider],
                    )

if __name__ == "__main__":
    demo.queue().launch(theme=theme, css=custom_css, share=False)

