import os, time, inspect
import gradio as gr
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODELS = []
for m in client.models.list():
    name = m.name.replace("models/", "")
    actions = getattr(m, "supported_actions", None) or []
    if actions and "generateContent" not in actions:
        continue
    if "flash" in name and not any(x in name for x in ["image", "tts", "live", "audio", "embedding"]):
        MODELS.append(name)
MODELS.sort(reverse=True)
if not MODELS:
    MODELS = ["gemini-3.8-flash"]

class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    explanation: str

class StudyOutput(BaseModel):
    summary_points: List[str]
    quiz: List[MCQ]

def study_buddy(text, num_questions=5, max_points=6):
    prompt = f"""You are a smart study assistant.
Read the study text below and produce:
1) summary_points: at most {max_points} short key points.
2) quiz: exactly {num_questions} multiple-choice questions, each with 4 options,
   one correct answer (correct_index from 0 to 3), and a one-sentence explanation.
Rules: base everything ONLY on the text. Write in the SAME language as the text.
Make wrong options plausible.

TEXT:
{text}"""
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StudyOutput, temperature=0.4))
                return r.parsed or StudyOutput.model_validate_json(r.text)
            except Exception as e:
                last_error = e
                if getattr(e, "code", None) in (429, 500, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
    raise RuntimeError(f"All attempts failed: {last_error}")

MAX_Q = 5
BTN_TEXT = "✨ لخّص واختبرني"

theme = gr.themes.Soft(primary_hue="purple", secondary_hue="fuchsia")
css = """
.gradio-container {
  background: radial-gradient(ellipse at top, #5b21b6 0%, #2e1065 50%, #0f0720 100%) !important;
}
.gradio-container::before, .gradio-container::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(2px 2px at 20px 30px, #ffffff, transparent),
    radial-gradient(2px 2px at 90px 140px, #e9d5ff, transparent),
    radial-gradient(1.5px 1.5px at 160px 70px, #ffffff, transparent),
    radial-gradient(2.5px 2.5px at 220px 190px, #f5d0fe, transparent),
    radial-gradient(1px 1px at 60px 210px, #ffffff, transparent);
  background-size: 250px 250px;
  animation: twinkle 3s ease-in-out infinite alternate;
}
.gradio-container::after {
  background-size: 340px 340px; background-position: 90px 60px;
  animation-duration: 5s; animation-delay: 1s; opacity: .7;
}
@keyframes twinkle { from { opacity: .25; } to { opacity: 1; } }
.gradio-container > * { position: relative; z-index: 1; }
#title, #title * { color: #f3e8ff !important; text-align: center;
  text-shadow: 0 0 12px #c084fc, 0 0 24px #a855f7; }
"""

if "css" in inspect.signature(gr.Blocks.launch).parameters:
    blocks_kwargs, launch_kwargs = {}, dict(theme=theme, css=css)
else:
    blocks_kwargs, launch_kwargs = dict(theme=theme, css=css), {}

def generate(text, n):
    hidden = [gr.update(choices=[], value=None, visible=False) for _ in range(MAX_Q)]
    if not text or not text.strip():
        yield ["⚠️ الصقي نصًا دراسيًا أولًا", None] + hidden + ["", gr.update(value=BTN_TEXT, interactive=True)]
        return
    yield ["⏳ **جاري تحليل النص وتوليد الملخص والأسئلة... انتظري قليلًا**", None] + hidden + ["", gr.update(value="⏳ جاري المعالجة...", interactive=False)]
    try:
        result = study_buddy(text, num_questions=int(n))
    except Exception as e:
        yield [f"❌ حصل خطأ: {str(e)[:300]}\n\nجرّبي مرة ثانية بعد قليل.", None] + hidden + ["", gr.update(value=BTN_TEXT, interactive=True)]
        return
    summary = "### 📝 الملخص الذكي\n\n" + "\n\n".join(f"• {p}" for p in result.summary_points)
    radios = []
    for i in range(MAX_Q):
        if i < len(result.quiz):
            q = result.quiz[i]
            radios.append(gr.update(choices=q.options, value=None, label=f"Q{i+1}. {q.question}", visible=True))
        else:
            radios.append(gr.update(choices=[], value=None, visible=False))
    yield [summary, result.model_dump()] + radios + ["", gr.update(value=BTN_TEXT, interactive=True)]

def check(quiz, *answers):
    if not quiz:
        return "ولّدي الأسئلة أولًا"
    score, lines = 0, []
    for i, q in enumerate(quiz["quiz"]):
        correct = q["options"][q["correct_index"]]
        ok = answers[i] == correct
        score += ok
        lines.append(f"{'✅' if ok else '❌'} **Q{i+1}:** {correct}\n\n💡 {q['explanation']}")
    return f"### 🎯 النتيجة: {score}/{len(quiz['quiz'])}\n\n" + "\n\n".join(lines)

with gr.Blocks(title="AI Smart Study Buddy", **blocks_kwargs) as demo:
    gr.Markdown("# ⭐ ✨ AI Smart Study Buddy ✨ ⭐\n🌟 الصق نصك الدراسي واحصل على ملخص واختبار فوري 🌟", elem_id="title")
    text = gr.Textbox(lines=8, label="النص الدراسي")
    n = gr.Slider(3, 5, value=5, step=1, label="عدد الأسئلة")
    btn = gr.Button(BTN_TEXT, variant="primary")
    summary = gr.Markdown()
    quiz_state = gr.State()
    radios = [gr.Radio(choices=[], visible=False, label="") for _ in range(MAX_Q)]
    check_btn = gr.Button("✅ صحّح إجاباتي")
    result_md = gr.Markdown()
    btn.click(generate, [text, n], [summary, quiz_state] + radios + [result_md, btn])
    check_btn.click(check, [quiz_state] + radios, result_md)

demo.launch(**launch_kwargs)
