import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Writing Coach"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

openai_api_key = st.secrets["OPENAI_API_KEY"]
model = ChatOpenAI(
#    model_name="gpt-3.5-turbo",
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=openai_api_key,
    tags=["wc-llm"],
)

# Writing Coach main prompt
WC_TEMPLATE = """
Your name is Writing Coach. You are a helpful friendly educator. Your goal is to evaluate the student's skills \
according to the specified Common Core learning standard using Student's interest topic as a basis. You must make sure \
all of your outputs are adapted for a general knowledge level of a 4th grader. 

Inputs:
Common Core standard: {standard}
Student's interest topic: {interest}
Previous data: {previous_data}
Student response: {student_response}

Your will generate outputs according to the instructions for the specific stage and stop.
You can generate outputs only for one stage depending on the inputs conditions, not both.  

If Previous data and Student response are NULL, run Stage 1 instructions.
<Stage1 instructions>
Stage 1 outputs: Intro, Rubric, Context, FRQ.
In Stage 1 you will introduce yourself (Intro), generate rubric that you will later use to assess student's \
response (Rubric), provide the necessary context (Context) and ask a Free Response Question (FRQ).

In Intro you must introduce yourself and clearly explain your goals.

Rubric must be based on the specified Common Core standard. It must consist of 4 criteria each graded on a scale \
from 0 to 5 with 20/20 as a maximum score for Final Grade. It must be in the form of a markdown table.

Context must provide specific factual data sufficient for student to answer the following FRQ. This section must not \
imply any additional external knowledge or suggestions to imagine knowledge. Context must include 200 words or more.

FRQ must solely rely upon the data provided in Context and must not require any additional knowledge.
</Stage1 instructions>

If Previous data and Student response are NOT NULL, run Stage 2 instructions. 
<Stage2 instructions>
Stage 2 outputs: Eval, Feedback.
In Stage 2 you will evaluate (Eval) the student's response according to the Rubric and provide useful and detailed \
feedback (Feedback).  

Eval must start with Final Grade (?/20) followed by grades by each criterion (?/5). It must be in the form of a \
markdown table.

Feedback must provide detailed analysis for each criterion in a form of a bullet-point list.
</Stage2 instructions>

Output only the JSON structure specified in the instruction below, DO NOT output anything except the JSON:
{format_instructions}
"""
wc_prompt = ChatPromptTemplate.from_template(WC_TEMPLATE)

input_standard = 'CCSS.ELA-LITERACY.W.4.9 - Draw evidence from literary or informational texts\
 to support analysis, reflection, and research.'
input_interest = 'baseball'
student_response = 'NULL'
previous_data = 'NULL'

st.title('👩‍🏫 Writing Coach')
st.sidebar.title("Writing Coach")

with st.sidebar:
    input_standard = st.text_area('Common Core Standard', input_standard)
    input_interest = st.text_input('Student Interest', input_interest)
    std_resp_type = st.radio("Which type of student response should we generate?", ["Right", "Wrong"])
    st.divider()
    if st.button("Clear All Cache"):
        st.cache_data.clear()

@st.cache_data
def wc_stage1(standard, interest):
    wc_chain = LLMChain(
        llm=model,
        prompt=wc_prompt,
        tags=["wc", "stage1"]
    )
    # Format and parser for Stage 1
    intro_schema = ResponseSchema(name='intro', description='')
    rubric_schema = ResponseSchema(name='rubric',
                                   description='Tell the student you will use the following \
                                    rubric to evaluate their answer and then state the rubric \
                                    succinctly in the form of a table')
    context_schema = ResponseSchema(name='context', description='')
    frq_schema = ResponseSchema(name='frq',
                                description='The FRQ to evaluate the student\'s skills')
    response_schemas = [intro_schema, rubric_schema, context_schema, frq_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    wc_response = wc_chain.run(
        standard=standard,
        interest=interest,
        previous_data='',
        student_response='',
        format_instructions=format_instructions,
    )
    return wc_response, output_parser.parse(wc_response)


with st.spinner('WC LLM Chain Stage 1...'):
    wc_st1_response, wc_output_dict = wc_stage1(input_standard, input_interest)

with st.chat_message("👩‍🏫"):
    st.write(wc_output_dict['intro'])
    st.info(wc_output_dict['rubric'])
    st.info(wc_output_dict['context'])
    st.divider()
    st.write(wc_output_dict['frq'])

# st.button('Rerun')

# Student response prompt
STD_TEMPLATE = """
You are a 4th grade student that has been given the following assignment:

{assignment}

You must give the {response_type} answer. Answer with 100 words or fewer.
"""
std_prompt = ChatPromptTemplate.from_template(STD_TEMPLATE)


@st.cache_data
def gen_std_resp(assignment, response_type):
    std_chain = LLMChain(
        llm=model,
        prompt=std_prompt,
        tags=["std"]
    )
    return std_chain.run(assignment=assignment, response_type=response_type)


with st.spinner('Student LLM Chain...'):
    student_response = gen_std_resp(wc_st1_response, std_resp_type)

with st.form('student_form'):
    student_response = st.text_area('Student response:', student_response, height=250, key='std_resp_input')
    std_resp_submitted = st.form_submit_button('Submit')

# Data generated at Stage 1
previous_data = wc_st1_response


@st.cache_data
def wc_stage2(student_response):
    wc_chain = LLMChain(
        llm=model,
        prompt=wc_prompt,
        tags=["wc", "stage2"]
    )
    # Format and parser for Stage 2
    eval_schema = ResponseSchema(name='eval',
                                 description='Evaluate the student response according to previously generated rubric. \
        Give score on each criteria and the final score.')
    feedback_schema = ResponseSchema(name='feedback', description='Provide student with detailed and useful feedback')
    response_schemas = [eval_schema, feedback_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    wc_response = wc_chain.run(
        standard=input_standard,
        interest=input_interest,
        previous_data=previous_data,
        student_response=student_response,
        format_instructions=format_instructions,
    )
    return wc_response, output_parser.parse(wc_response)


if std_resp_submitted:
    with st.spinner('WC LLM Chain Stage 2...'):
        wc_st2_response, wc2_output_dict = wc_stage2(student_response)
    with st.chat_message("👩‍🏫"):
        st.info(wc2_output_dict['eval'])
        st.write(wc2_output_dict['feedback'])
