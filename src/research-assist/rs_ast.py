import json
from xact.llm import llm_client
from xact.settings import config
from xact.utils import log
from xact.llm.llm import LLM
from xact.plugin.doc import pdf
import chromadb

log.toggle_logging(True)
config.XACT_LLM_MODEL = "qwen2.5:1.5b"
pdfobj = pdf.PDF()

client = chromadb.PersistentClient(
    path="vec-db",
)
col = client.get_or_create_collection("pdfdoc")



def db_chat(prompt):
    model = "qwen2.5:1.5b"
    db_res = col.query(query_texts=prompt.strip())
    documents = db_res["documents"][0]
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are given with a documents asnwer any question with the given documents if it has relevent content",
            },
            {
                "role": "user",
                "content": f"""documents : {json.dumps(documents)} \n question : {prompt.strip()}""",
            },
        ],
        model=model,
        temperature=0,
    )
    log.logger.info("llm out generated")
    return chat_completion.choices[0].message.content



def add_db(pdf_path):

    pdfchunks = pdfobj.read(pdf_path)

    for doc in pdfchunks:
        col.upsert(
            ids=doc.id,
            documents=doc.content,
            metadatas=doc.meta_data,
        )



# add_db(pdf_path="/home/lokesh/Downloads/Lokeshwaran-RESUME.pdf")
# db_chat("what are the lokesh skills")