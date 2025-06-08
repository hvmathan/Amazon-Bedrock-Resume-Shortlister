import streamlit as st
import boto3
import json
import re
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

# Constants
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
REGION = "us-east-1"

job_descriptions = {
    "Project Manager": """We are seeking a results-oriented Project Manager with 5+ years of experience managing cross-functional teams in agile environments. 
The candidate should have excellent communication skills, experience with tools like Jira and Confluence, and a proven ability to deliver projects on time and within budget.
Note: Candidates located in Hyderabad, India and women are especially encouraged to apply, as this role aims to improve gender representation in leadership.""",

    "Software Developer": """We are looking for a Full Stack Software Developer with 3+ years of experience in front-end frameworks (React, Angular) and back-end technologies (Node.js, Python, or Java).
Experience with RESTful APIs, cloud platforms (AWS, Azure), and CI/CD pipelines is highly desirable. The role is remote-friendly and open to diverse candidates from all locations.""",

    "Intern": """We are looking for a motivated and quick-learning Intern to assist in software testing, documentation, and market research. 
Strong communication and a willingness to explore new tools are key. This is a 3-month internship with mentorship from senior staff.""",

    "Team Lead": """We are hiring a Team Lead to mentor engineers, coordinate sprints, and ensure high-quality deliverables.
Ideal candidates should have 6+ years of software development experience and at least 1 year of people management. Strong leadership and technical skills are a must.""",

    "HR Manager": """We are seeking an HR Manager with experience in talent acquisition, employee engagement, and HR compliance.
Candidates should be familiar with labor laws, performance appraisal systems, and DEI (Diversity, Equity & Inclusion) best practices. Hybrid role based in Bangalore.""",

    "Sales Expert": """We need a Sales Expert with a proven record in B2B SaaS sales, excellent negotiation skills, and the ability to manage key accounts. 
This is a target-driven role requiring travel across India. Candidates from any gender or location are welcome; multilingual skills are a plus."""
}

# --- Helper Functions ---
def extract_resume_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

def build_prompt(resume_text, jd_text):
    return f"""
Human: Evaluate the following resume against this job description. Score the match out of 100 and explain why.

Job Description:
{jd_text}

Resume:
{resume_text}

Return a JSON object with:
- "score": integer
- "reasoning": list of 2-3 sentences
- "missing": list of skills or experiences missing
Respond only with valid JSON.
Assistant:
"""

def get_bedrock_response(prompt):
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.3
    }
    try:
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except (BotoCoreError, ClientError) as e:
        st.error(f"Error from Bedrock: {e}")
        return None

def extract_json(raw_output):
    try:
        match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"❌ JSON parsing failed: {e}")
        st.text_area("🔍 Raw Bedrock Output", raw_output, height=200)
        return None

# --- Streamlit UI ---
logo = Image.open("IITK_logo.jpg")
st.image(logo, width=120)  
st.title("📄 AI Resume Shortlister & Analytics Dashboard")

role = st.selectbox("Select Job Role", list(job_descriptions.keys()))
uploaded_files = st.file_uploader("Upload Resumes", type=["docx"], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"📑 Evaluating: `{uploaded_file.name}`")
        resume_text = extract_resume_text(uploaded_file)
        job_description = job_descriptions.get(role, "")
        prompt = build_prompt(resume_text, job_description)

        with st.spinner("Sending to Amazon Bedrock..."):
            raw_output = get_bedrock_response(prompt)

        if raw_output:
            parsed = extract_json(raw_output)
            if parsed:
                score = parsed.get("score", 0)
                reasoning = " | ".join(parsed.get("reasoning", []))
                missing = parsed.get("missing", [])
                results.append({
                    "name": uploaded_file.name,
                    "score": score,
                    "reasoning": reasoning,
                    "missing": missing
                })

                st.success(f"✅ Score: {score}")
                st.write(f"💡 Reasoning: {reasoning}")
                if missing:
                    st.warning(f"❌ Missing Skills: {' | '.join(missing)}")
        else:
            st.error("⚠️ No response from Bedrock.")

# --- Analytics Dashboard ---
if results:
    st.subheader("📊 Resume Analytics")

    df = pd.DataFrame(results)

    # Score Distribution
    st.markdown("### 📈 Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["score"], bins=10, color="skyblue", edgecolor="black")
    ax.set_xlabel("Match Score")
    ax.set_ylabel("Number of Candidates")
    st.pyplot(fig)

    # Top Candidates
    st.markdown("### 🏆 Top 5 Candidates")
    top_candidates = df.sort_values(by="score", ascending=False).head(5)
    st.dataframe(top_candidates[["name", "score", "reasoning"]])

    # Uncomment this section if you want WordCloud
    # st.markdown("### ☁️ Missing Skills WordCloud")
    # all_missing = [skill for sublist in df["missing"] for skill in sublist]
    # if all_missing:
    #     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_missing))
    #     fig_wc, ax_wc = plt.subplots()
    #     ax_wc.imshow(wordcloud, interpolation='bilinear')
    #     ax_wc.axis("off")
    #     st.pyplot(fig_wc)
    # else:
    #     st.info("✅ No missing skills across uploaded resumes!")

    # Download results
    st.download_button("⬇️ Download Results CSV", data=df.to_csv(index=False), file_name="resume_evaluation_results.csv")
