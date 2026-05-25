import gradio as gr
from rag_engine import build_rag_chain, ask_question

qa_chain = None

def process_pdf(pdf_file):
    global qa_chain
    if pdf_file is None:
        return "Please upload a PDF first."
    try:
        qa_chain = build_rag_chain(pdf_file.name)
        return "PDF processed! You can now ask questions."
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(question):
    global qa_chain
    if qa_chain is None:
        return "Please upload and process a PDF first."
    if not question.strip():
        return "Please enter a question."
    try:
        result = ask_question(qa_chain, question)
        answer = result["answer"]
        sources = "\n\n---\n".join(result["sources"])
        return f"**Answer:**\n{answer}"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="RAG Document Q&A") as app:
    gr.Markdown("#RAG Document Q&A Assistant")
    gr.Markdown("Upload any PDF and ask questions about it in plain English.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Process PDF", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is this document about?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Markdown(label="Answer")

    upload_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status)
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    app.launch()