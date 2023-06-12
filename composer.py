import gradio as gr
import numpy as np

class HumComposer:
    def __init__(self, generate, generate_with_chroma):
        self.generate = generate
        self.generate_with_chroma = generate_with_chroma
    
    def __call__(self, melody, description):
        if melody:
            generated = self.generate_with_chroma(description, melody[1].astype(np.float32), melody[0])
        else:
            generated = self.generate(description)
        
        return generated


def create_interface(composer: HumComposer):
    block = gr.Blocks(css=".gradio-container")

    with block:
        with gr.Row():
            gr.Markdown("<h1><center>Hum Composer with BentoML 🍱</center></h1><br/><center>Compose a musical masterpiece, inspired by the melody in your mind.</center>")
        with gr.Row():
            melody_input = gr.Audio(label="Hum a melody line", source="microphone")
            description_input = gr.Text(label="Describe the style of the music to be generated based on the melody", value="Funky jazz")
        with gr.Row():
            generate_button = gr.Button(value="Generate")
        with gr.Row():
            music_output = gr.Audio(label="Generated music")
    
        generate_button.click(
            composer,
            inputs=[melody_input, description_input],
            outputs=[music_output],
        )

    return block
