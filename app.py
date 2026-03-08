import gradio as gr
from agentforge_env.server.app import app
from demo.app import build_demo

gr.mount_gradio_app(app, build_demo(), path="/web")
