import gradio as gr
from chatbot import AIChatbot

bot = AIChatbot()

def respond(message, history):
    response = bot.generate_response(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="🤖 AI Chatbot",
    description="Ask me anything!",
    textbox=gr.Textbox(placeholder="Type your message here...", scale=7),
    retry_btn=None,
    undo_btn=None,
    clear_btn="🗑️ Clear Chat"
)

demo.launch(server_port=7860)