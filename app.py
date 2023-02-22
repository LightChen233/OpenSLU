'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-07 15:42:32
LastEditTime: 2023-02-20 09:41:27
Description: 

'''
import argparse
import gradio as gr

from common.config import Config
from common.model_manager import ModelManager
from common.utils import str2bool


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', '-cp', type=str, default="config/examples/from_pretrained.yaml")
parser.add_argument('--push_to_public', '-p', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Push to public network.")
args = parser.parse_args()
config = Config.load_from_yaml(args.config_path)
config.base["train"] = False
config.base["test"] = False

model_manager = ModelManager(config)
model_manager.init_model()


def text_analysis(text):
    print(text)
    data = model_manager.predict(text)
    html = """<link href="https://cdn.staticfile.org/twitter-bootstrap/5.1.1/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.staticfile.org/twitter-bootstrap/5.1.1/js/bootstrap.bundle.min.js"></script>"""
    html += """<div style="background: white; padding: 16px;"><b>Intent:</b>"""

    for intent in data["intent"]:
        html += """<button type="button" class="btn btn-white">
                        <span class="badge text-dark btn-light">""" + intent + """</span> </button>"""
    html += """<br /> <b>Slot:</b>"""
    for t, slot in zip(data["text"], data["slot"]):
        html += """<button type="button" class="btn btn-white">"""+t+"""<span class="badge text-dark" style="background-color: rgb(255, 255, 255);
                            color: rgb(62 62 62);
                            box-shadow: 2px 2px 7px 1px rgba(210, 210, 210, 0.42);">"""+slot+\
                            """</span>
                    </button>"""
    html+="</div>"
    return html


demo = gr.Interface(
    text_analysis,
    gr.Textbox(placeholder="Enter sentence here..."),
    ["html"],
    examples=[
        ["i would like to find a flight from charlotte to las vegas that makes a stop in st louis"],
    ],
)
if args.push_to_public:
    demo.launch(share=True)
else:
    demo.launch()
