# Amazon-Bedrock-Resume-Shortlister

Recently, I came across a situation to fill a position for my team. The moment we opened the role, I received close to 300 resumes for evaluation. While our HR team did their best to manually sift through them based on job description, experience, skill set, and other factors... I realized:
Why not automate this using Generative AI?

And so, this product was born :)

What This Is About
This project explores how Generative AI (via Amazon Bedrock) can help HR teams:

Automatically screen resumes
Evaluate candidate fitment to job descriptions
Provide actionable insights
All in minutes, not days.
The Problem
Recruiters often face:

Hundreds of resumes per job posting
Time-consuming manual screening
Inconsistent and subjective evaluation
The Solution
With the power of Amazon Bedrock + Python + Streamlit, I built a Resume Shortlister that:

Takes resumes (.docx) as input
Evaluates them against a selected job description
Returns:
Match Score
Reasoning
Missing Skills
Tech Stack
Amazon Bedrock (Titan Text - Nova)
Python
Streamlit for UI
Matplotlib, WordCloud for analytics
Pandas for data handling
Boto3 for AWS integration
Folder Structure
Image description

How It Works (Code Snippet on a high level)
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
resume_shortlister.py

import boto3
import json
from docx import Document

# ---------- Step 1: Load Resume ----------
def extract_resume_text(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# ---------- Step 2: Sample Job Descriptions ----------
job_descriptions = {
    "Project Manager": """
We are looking for an experienced Project Manager to lead cross-functional teams...
(Key skills: Agile, Scrum, Budgeting, JIRA, Communication)
""",
    "Software Developer": """
Join our dev team to build scalable backend services...
(Key skills: Python, Java, APIs, AWS, CI/CD)
"""
}

# ---------- Step 3: Build Prompt ----------
def build_prompt(jd_text, resume_text):
    return f"""
You are an AI assistant helping shortlist job candidates.

Here is the job description:
{jd_text}

Here is the resume:
{resume_text}

Task:
1. Rate this resume from 0 to 100 based on its relevance to the job.
2. List the top 3 reasons why this resume is a good or poor fit.
3. Mention any important missing qualifications.

Respond in JSON format:
{{
  "score": <score>,
  "reasoning": ["..."],
  "missing": ["..."]
}}
"""

# ---------- Step 4: Call Bedrock ----------
def call_bedrock(prompt):
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        body=json.dumps(body),
        contentType="application/json",
)


    output = json.loads(response['body'].read().decode())
    return output['content'][0]['text']

# ---------- Run ----------
if __name__ == "__main__":
    resume_path = "resumes/Project_Manager_1.docx"
    role = "Project Manager"  # change role

    resume_text = extract_resume_text(resume_path)
    jd = job_descriptions[role]
    prompt = build_prompt(jd, resume_text)

    print("\n📤 Sending prompt to Bedrock...\n")
    result = call_bedrock(prompt)

    print("✅ Response from Bedrock:\n")
    print(result)
streamlit_app.py

import streamlit as st
import os
import boto3
import json
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from botocore.exceptions import BotoCoreError, ClientError

# Constants
BEDROCK_MODEL_ID = "amazon.titan-text-lite-v1"
REGION = "us-east-1"

job_descriptions = {
    "Project Manager": "We are looking for an experienced Project Manager to lead cross-functional teams...",
    "Software Developer": "We are looking for a Software Developer with experience in front-end and back-end technologies...",
    "Intern": "We are looking for a motivated Intern with a strong willingness to learn and assist in various tasks...",
    "Team Lead": "Looking for a Team Lead to guide and support engineering teams in agile delivery...",
    "HR Manager": "We need a skilled HR Manager to handle talent acquisition, employee engagement, and compliance...",
    "Sales Expert": "Seeking a sales expert with experience in B2B and client relationship management..."
    # Add more roles as needed
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
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

# --- Streamlit UI ---
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
            try:
                parsed = json.loads(raw_output)
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
            except json.JSONDecodeError:
                st.error("❌ Failed to parse Bedrock's response.")
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

    # WordCloud of Missing Skills
    st.markdown("### ☁️ Missing Skills WordCloud")
    all_missing = [skill for sublist in df["missing"] for skill in sublist]
    if all_missing:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_missing))
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("✅ No missing skills across uploaded resumes!")


Streamlit Demo (UI Preview)
Image description

Once the above command is run, it opens up the UI which has the below features :

Upload multiple resumes
Choose Job Role (Project Manager, Developer, Intern, etc.)
See Match Score, Reasoning, and Missing Skills
View charts
Image description

Suppose I'm hiring a Software Developer for my team — I simply select the relevant role from the dropdown menu.

Image description

Next, I upload the resumes I’ve received. For demonstration purposes, I’ve included a mix of profiles to test whether Bedrock can accurately identify the most relevant candidates.

Image description

Bedrock now takes over and begins evaluating each profile.
If you're curious about what exactly we're sending to Bedrock, you can revisit the Python code — The prompt is designed to be clear and structured. Bedrock then uses Amazon Titan (Nova) under the hood to evaluate the resume and return results.

Once the analysis is complete, we get a structured summary for each resume, including:

Match Score (out of 100)

Reasoning behind the score

Missing skills or experience compared to the job description

This gives instant clarity to recruiters on which candidates are worth shortlisting — all without manually opening a single resume.

Image description

{
"score": 85,
"reasoning": [
"The resume demonstrates relevant experience as a project manager...",
"The candidate has 1 year of experience...",
"Highlights skills in team coordination..."
],
"missing": [
"No mention of JIRA",
"Education section lacks detail"
]
}

Finally, the system presents the Top 5 candidates — each with a match score and clear reasoning — making it incredibly easy for the HR team to not just identify qualified profiles, but to confidently select the best of the best.

Image description

Impact on HR Teams
Saves time and effort
Transparent and explainable results
Consistent, AI-assisted decision making
Shortlist ready-to-interview candidates faster
Final Thoughts
This project showcases how Generative AI combined with Amazon Nova can revolutionize Talent Acquisition and HR Analytics — automating resume screening, improving decision-making, and saving hours of manual effort.

Whether you're an HR professional, a hiring manager, or an AI enthusiast, this is a glimpse into the future of intelligent hiring.
